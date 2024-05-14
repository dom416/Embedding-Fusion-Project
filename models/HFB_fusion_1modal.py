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

        self.gene_in = input_dims  

        self.gene_hidden = hidden_dims[0]
        self.cox_hidden = hidden_dims[1]

        self.output_intra = output_dims[0]
        self.output_inter = output_dims[1]
        self.label_dim = output_dims[2]
        self.rank = rank
        self.factor_drop = fac_drop

        self.gene_prob = dropouts[0]
        
        self.cox_prob = dropouts[1]

        self.joint_output_intra = self.rank * self.output_intra
        self.joint_output_inter = self.rank * self.output_inter
        self.in_size = self.gene_hidden + self.output_intra + self.output_inter
        self.hid_size = self.gene_hidden


        self.norm = nn.BatchNorm1d(self.in_size)
        self.factor_drop = nn.Dropout(self.factor_drop)
        self.attention = nn.Sequential(nn.Linear((self.hid_size + self.output_intra), 1), nn.Sigmoid())

        self.encoder_gene = SubNet(self.gene_in, self.gene_hidden)

        self.Linear_gene = nn.Linear(self.gene_hidden, self.joint_output_intra)

        self.Linear_gene_a = nn.Linear(self.gene_hidden + self.output_intra, self.joint_output_inter)

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

    def forward(self, x1):
        gene_feature = self.encoder_gene(x1.squeeze(1))

        gene_h = self.Linear_gene(gene_feature)

        ######################### modelity-specific###############################
        #intra_interaction#
        intra_gene = self.mfb(gene_h, gene_h, self.output_intra)

        gene_x = torch.cat((gene_feature, intra_gene), 1)

        sg = self.attention(gene_x)

        sg_a = (sg.expand(gene_feature.size(0), (self.gene_hidden + self.output_intra)))

        gene_x_a = sg_a * gene_x

        unimodal = gene_x_a

        ######################### cross-modelity######################################
        
        ############################################### fusion layer ###################################################

        fusion = unimodal
        fusion = self.norm(fusion)
        code = self.encoder(fusion)
        out = self.classifier(code)
        out = out * self.output_range + self.output_shift
        return out, code

