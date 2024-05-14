import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SubNet, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh())
        encoder2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.encoder = nn.Sequential(encoder1, encoder2)
    def forward(self, x):
        y = self.encoder(x)
        return y


class HFBSurv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, dropouts, rank,fac_drop):
        super(HFBSurv, self).__init__()

        self.clin_in = input_dims[0]
        self.pathr_in = input_dims[1]
        self.unis_in = input_dims[2]
        self.ct_in = input_dims[3]
        self.remedis_in = input_dims[4]

        self.clin_hidden = hidden_dims[0]
        self.pathr_hidden = hidden_dims[1]
        self.unis_hidden = hidden_dims[2]
        self.ct_hidden = hidden_dims[3]
        self.remedis_hidden = hidden_dims[4]
        self.cox_hidden = hidden_dims[5]

        self.output_intra = output_dims[0]
        self.output_inter = output_dims[1]
        self.label_dim = output_dims[2]
        self.rank = rank
        self.factor_drop = fac_drop

        self.clin_prob = dropouts[0]
        self.pathr_prob = dropouts[1]
        self.unis_prob = dropouts[2]
        self.ct_prob = dropouts[3]
        self.remedis_prob = dropouts[4]
        self.cox_prob = dropouts[5]

        self.joint_output_intra = self.rank * self.output_intra
        self.joint_output_inter = self.rank * self.output_inter
        self.in_size = self.clin_hidden + self.output_intra + self.output_inter
        self.hid_size = self.clin_hidden

        self.norm = nn.BatchNorm1d(self.in_size)
        self.factor_drop = nn.Dropout(self.factor_drop)
        self.attention = nn.Sequential(nn.Linear((self.hid_size + self.output_intra), 1), nn.Sigmoid())

        self.encoder_clin = SubNet(self.clin_in, self.clin_hidden)
        self.encoder_pathr = SubNet(self.pathr_in, self.pathr_hidden)
        self.encoder_unis = SubNet(self.unis_in, self.unis_hidden)
        self.encoder_ct= SubNet(self.ct_in, self.ct_hidden)
        self.encoder_remedis = SubNet(self.remedis_in, self.remedis_hidden)

        self.Linear_clin = nn.Linear(self.clin_hidden, self.joint_output_intra)
        self.Linear_pathr = nn.Linear(self.pathr_hidden, self.joint_output_intra)
        self.Linear_unis = nn.Linear(self.unis_hidden, self.joint_output_intra)
        self.Linear_ct = nn.Linear(self.ct_hidden, self.joint_output_intra)
        self.Linear_remedis = nn.Linear(self.remedis_hidden, self.joint_output_intra)

        self.Linear_clin_a = nn.Linear(self.clin_hidden + self.output_intra, self.joint_output_inter)
        self.Linear_pathr_a = nn.Linear(self.pathr_hidden + self.output_intra, self.joint_output_inter)
        self.Linear_unis_a = nn.Linear(self.unis_hidden + self.output_intra, self.joint_output_inter)
        self.Linear_ct_a = nn.Linear(self.ct_hidden + self.output_intra, self.joint_output_inter)
        self.Linear_remedis_a = nn.Linear(self.remedis_hidden + self.output_intra, self.joint_output_inter)

        #########################the layers of survival prediction#####################################
        encoder1 = nn.Sequential(nn.Linear(self.in_size, self.cox_hidden), nn.Tanh(), nn.Dropout(p=self.cox_prob))
        encoder2 = nn.Sequential(nn.Linear(self.cox_hidden, 64), nn.Tanh(), nn.Dropout(p=self.cox_prob))
        self.encoder = nn.Sequential(encoder1, encoder2)
        self.classifier = nn.Sequential(nn.Linear(64, self.label_dim), nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def mfb(self, x1, x2, output_dim):

        self.output_dim =  output_dim
        fusion = torch.mul(x1, x2)
        fusion = self.factor_drop(fusion)
        fusion = fusion.view(-1, 1, self.output_dim, self.rank)
        fusion = torch.squeeze(torch.sum(fusion, 3))
        fusion = torch.sqrt(F.relu(fusion)) - torch.sqrt(F.relu(-fusion))
        fusion = F.normalize(fusion)
        return fusion

    def forward(self, x1, x2, x3, x4, x5):
        clin_feature = self.encoder_clin(x1.squeeze(1))
        pathr_feature = self.encoder_pathr(x2.squeeze(1))
        unis_feature = self.encoder_unis(x3.squeeze(1))
        ct_feature = self.encoder_ct(x4.squeeze(1))
        remedis_feature = self.encoder_remedis(x5.squeeze(1))

        clin_h = self.Linear_clin(clin_feature)
        pathr_h = self.Linear_pathr(pathr_feature)
        unis_h = self.Linear_unis(unis_feature)
        ct_h = self.Linear_ct(ct_feature)
        remedis_h = self.Linear_remedis(remedis_feature)

        ######################### modelity-specific###############################
        #intra_interaction#
        intra_clin = self.mfb(clin_h, clin_h, self.output_intra)
        intra_pathr = self.mfb(pathr_h, pathr_h, self.output_intra)
        intra_unis = self.mfb(unis_h, unis_h, self.output_intra)
        intra_ct = self.mfb(ct_h, ct_h, self.output_intra)
        intra_remedis = self.mfb(remedis_h, remedis_h, self.output_intra)

        clin_x = torch.cat((clin_feature, intra_clin), 1)
        pathr_x = torch.cat((pathr_feature, intra_pathr), 1)
        unis_x = torch.cat((unis_feature, intra_unis), 1)
        ct_x = torch.cat((ct_feature, intra_ct), 1)
        remedis_x = torch.cat((remedis_feature, intra_remedis), 1)

        sclin = self.attention(clin_x)
        spathr = self.attention(pathr_x)
        sunis = self.attention(unis_x)
        sct = self.attention(ct_x)
        sremedis = self.attention(remedis_x)

        sclin_a = (sclin.expand(clin_feature.size(0), (self.clin_hidden + self.output_intra)))
        spathr_a = (spathr.expand(pathr_feature.size(0), (self.pathr_hidden + self.output_intra)))
        sunis_a = (sunis.expand(unis_feature.size(0), (self.unis_hidden + self.output_intra)))
        sct_a = (sct.expand(ct_feature.size(0), (self.ct_hidden + self.output_intra)))
        sremedis_a = (sremedis.expand(remedis_feature.size(0), (self.remedis_hidden + self.output_intra)))

        clin_x_a = sclin_a * clin_x
        pathr_x_a = spathr_a * pathr_x
        unis_x_a = sunis_a * unis_x
        ct_x_a = sct_a * ct_x
        remedis_x_a = sremedis_a * remedis_x

        unimodal = clin_x_a + pathr_x_a + unis_x_a + ct_x_a + remedis_x_a

        ######################### cross-modelity######################################
        clin = F.softmax(clin_x_a, 1)
        pathr = F.softmax(pathr_x_a, 1)
        unis = F.softmax(unis_x_a, 1)
        ct = F.softmax(ct_x_a, 1)
        remedis = F.softmax(remedis_x_a, 1)

        sclin = sclin.squeeze()
        spathr = spathr.squeeze()
        sunis = sunis.squeeze()
        sct = sct.squeeze()
        sremedis = sremedis.squeeze()

        sclinpathr = (1 / (torch.bmm(clin.unsqueeze(1), pathr.unsqueeze(2)).squeeze() + 0.5) * (sclin + spathr))
        sclinunis = (1 / (torch.bmm(clin.unsqueeze(1), unis.unsqueeze(2)).squeeze() + 0.5) * (sclin + sunis))
        sclinct = (1 / (torch.bmm(clin.unsqueeze(1), ct.unsqueeze(2)).squeeze() + 0.5) * (sclin + sct))
        sclinremedis = (1 / (torch.bmm(clin.unsqueeze(1), remedis.unsqueeze(2)).squeeze() + 0.5) * (sclin + sremedis))
        spathrunis = (1 / (torch.bmm(pathr.unsqueeze(1), unis.unsqueeze(2)).squeeze() + 0.5) * (spathr + sunis))
        spathrct = (1 / (torch.bmm(pathr.unsqueeze(1), ct.unsqueeze(2)).squeeze() + 0.5) * (spathr + sct))
        spathrremedis = (1 / (torch.bmm(pathr.unsqueeze(1), remedis.unsqueeze(2)).squeeze() + 0.5) * (spathr + sremedis))
        sunisct = (1 / (torch.bmm(unis.unsqueeze(1), ct.unsqueeze(2)).squeeze() + 0.5) * (sunis + sct))
        sunisremedis = (1 / (torch.bmm(unis.unsqueeze(1), remedis.unsqueeze(2)).squeeze() + 0.5) * (sunis + sremedis))
        sctremedis = (1 / (torch.bmm(ct.unsqueeze(1), remedis.unsqueeze(2)).squeeze() + 0.5) * (sct + sremedis))


        normalize = torch.cat((sclinpathr.unsqueeze(1), sclinunis.unsqueeze(1), sclinct.unsqueeze(1), sclinremedis.unsqueeze(1), spathrunis.unsqueeze(1), spathrct.unsqueeze(1),               spathrremedis.unsqueeze(1), sunisct.unsqueeze(1), sunisremedis.unsqueeze(1), sctremedis.unsqueeze(1)), 1)
        normalize = F.softmax(normalize, 1)
        sclinpathr_a = normalize[:, 0].unsqueeze(1).expand(clin_feature.size(0), self.output_inter)
        sclinunis_a = normalize[:, 1].unsqueeze(1).expand(clin_feature.size(0), self.output_inter)
        sclinct_a = normalize[:, 2].unsqueeze(1).expand(clin_feature.size(0), self.output_inter)
        sclinremedis_a = normalize[:, 3].unsqueeze(1).expand(clin_feature.size(0), self.output_inter)
        spathrunis_a = normalize[:, 4].unsqueeze(1).expand(pathr_feature.size(0), self.output_inter)
        spathrct_a = normalize[:, 5].unsqueeze(1).expand(pathr_feature.size(0), self.output_inter)
        spathrremedis_a = normalize[:, 6].unsqueeze(1).expand(pathr_feature.size(0), self.output_inter)
        sunisct_a = normalize[:, 7].unsqueeze(1).expand(unis_feature.size(0), self.output_inter)
        sunisremedis_a = normalize[:, 8].unsqueeze(1).expand(unis_feature.size(0), self.output_inter)
        sctremedis_a = normalize[:, 9].unsqueeze(1).expand(ct_feature.size(0), self.output_inter)

        # inter_interaction#
        clin_l = self.Linear_clin_a(clin_x_a)
        pathr_l = self.Linear_pathr_a(pathr_x_a)
        unis_l = self.Linear_unis_a(unis_x_a)
        ct_l = self.Linear_ct_a(ct_x_a)
        remedis_l = self.Linear_remedis_a(remedis_x_a)

        inter_clin_pathr = self.mfb(clin_l, pathr_l, self.output_inter)
        inter_clin_unis = self.mfb(clin_l, unis_l, self.output_inter)
        inter_clin_ct = self.mfb(clin_l, ct_l, self.output_inter)
        inter_clin_remedis = self.mfb(clin_l, remedis_l, self.output_inter)
        inter_pathr_unis = self.mfb(pathr_l, unis_l, self.output_inter)
        inter_pathr_ct = self.mfb(pathr_l, ct_l, self.output_inter)
        inter_pathr_remedis = self.mfb(pathr_l, remedis_l, self.output_inter)
        inter_unis_ct = self.mfb(unis_l, ct_l, self.output_inter)
        inter_unis_remedis = self.mfb(unis_l, remedis_l, self.output_inter)
        inter_ct_remedis = self.mfb(ct_l, remedis_l, self.output_inter)

        bimodal = sclinpathr_a * inter_clin_pathr + sclinunis_a * inter_clin_unis + sclinct_a * inter_clin_ct +  sclinremedis_a*inter_clin_remedis  +  spathrunis_a*inter_pathr_unis  +  spathrct_a*inter_pathr_ct  + spathrremedis_a*inter_pathr_remedis + sunisct_a*inter_unis_ct  + sunisremedis_a*inter_unis_remedis + sctremedis_a*inter_ct_remedis
        ############################################### fusion layer ###################################################

        fusion = torch.cat((unimodal, bimodal), 1)
        fusion = self.norm(fusion)
        code = self.encoder(fusion)
        out = self.classifier(code)
        out = out * self.output_range + self.output_shift
        return out, code

