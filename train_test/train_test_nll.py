import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from HFB_fusion import HFBSurv
from torch.utils.data import DataLoader
from survival_dataset_5fold import SurvivalDataset
from utils import CoxLoss, regularize_weights, CIndex_lifeline, cox_log_rank, accuracy_cox,count_parameters
import torch.optim as optim
import pickle
import os
import gc
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from nll_loss_func import NLLSurvLoss
from discrete_hazards_plot import plot_survival_probabilities

####################################################### MFB ############################################################

def train(opt,train_data,test_data,device):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)
    model = HFBSurv((1024, 1024, 1024), (48, 48, 48, 256), (20, 20, 20), (0.1, 0.1, 0.1, 0.3), 20, 0.1).to(device)
    model.to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    # batch_size=int(len(train_data)/10)
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    c_index_best = 0

    for epoch in tqdm(range(opt.epoch_count, opt.niter+opt.niter_decay+1)):
        model.train()
        risk_pred_all, censor_all, survtime_all, case_id_all = np.array([]), np.array([]), np.array([]), np.array([])

        # added in for nll
        all_risk_scores = []
        all_censorships = []
        all_event_times = []

        loss_epoch = 0
        gc.collect()
        for batch_idx, batch in enumerate(train_loader):
            censor = batch['censor'].to(device)
            survtime = batch['survival_time'].to(device)
            x_clin = batch['clinical_data'].to(device)
            x_pathr = batch['pathology_report'].to(device)
            x_unis = batch['uni_slide_image'].to(device)
            true_time_bin = batch['true_time_bin'].to(device)
            print(censor)
          
            pred = model(x_clin, x_pathr, x_unis)
            if isinstance(pred, tuple):
              pred = pred[0]  # Extract the tensor if it's part of a tuple
            else:
              pred = pred  # Use directly if it's not a tuple
        
            # Ensure pred is on the correct device and is a tensor
            pred = pred.to(device)
            
            loss_fn = NLLSurvLoss(alpha=0)
            loss = loss_fn(h=pred, y=true_time_bin, t=survtime, c=censor)
            loss_reg = regularize_weights(model=model)
            loss = loss + opt.lambda_reg *  loss_reg

            # part for nll loss from PORPOISE:
            hazards = torch.sigmoid(pred)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = 1-torch.sum(survival, dim=1).detach().cpu().numpy()
            #risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            all_risk_scores.append(risk)
            all_censorships.append(censor.detach().cpu().numpy())
            all_event_times.append(survtime.detach().cpu().numpy())


            loss_epoch += loss.data.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_epoch /= len(train_loader.dataset)

            all_risk_scores = np.concatenate(all_risk_scores)
            all_censorships = np.concatenate(all_censorships)
            all_event_times = np.concatenate(all_event_times)

            cindex_epoch = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

            #loss_test, cindex_test = test(opt, model, test_data, device)

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
         

            #metric_logger['test']['loss'].append(loss_test)
            #metric_logger['test']['cindex'].append(cindex_test)
           

            #if cindex_test > c_index_best:
               # c_index_best = cindex_test
            #if opt.verbose > 0:
                #print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
               # print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))
        
    return model, optimizer, metric_logger

def test(opt,model, test_data, device):
    model.eval()
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]),np.array([])
    loss_test = 0
    code_final = None

    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, batch in enumerate(test_loader):
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        x_clin = batch['clinical_data'].to(device)
        x_pathr = batch['pathology_report'].to(device)
        x_unis = batch['uni_slide_image'].to(device)
        true_time_bin = batch['true_time_bin'].to(device)
        
        pred = model(x_clin, x_pathr, x_unis)
        
        if isinstance(pred, tuple):
            pred = pred[0]  # Extract the tensor if it's part of a tuple
        else:
            pred = pred  # Use directly if it's not a tuple
        
        # Ensure pred is on the correct device and is a tensor
        pred = pred.to(device)
        
        loss_fn = NLLSurvLoss(alpha=0)
        loss = loss_fn(h=pred, y=true_time_bin, t=survtime, c=censor)
        loss_test += loss.data.item()

        # part for nll loss from PORPOISE:
        hazards = torch.sigmoid(pred)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = 1-torch.sum(survival, dim=1).detach().cpu().numpy()
        #risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        all_risk_scores.append(risk)
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(survtime.detach().cpu().numpy())

        
        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
    
    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    cindex_test = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    #pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    #surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    #pred_test = [risk_pred_all, survtime_all, censor_all]
    #code_final_data = code_final.data.cpu().numpy()
    return loss_test, cindex_test
    
    
def test_and_plot(opt,model, test_data, device):
    model.eval()
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]),np.array([])
    loss_test = 0
    code_final = None

    all_risk_scores = []
    all_survival_scores = []
    all_censorships = []
    all_event_times = []
    all_case_ids = []

    for batch_idx, batch in enumerate(test_loader):
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        x_clin = batch['clinical_data'].to(device)
        x_pathr = batch['pathology_report'].to(device)
        x_unis = batch['uni_slide_image'].to(device)
        true_time_bin = batch['true_time_bin'].to(device)
        case_id = batch['case_id']
        
        pred = model(x_clin, x_pathr, x_unis)
        
        if isinstance(pred, tuple):
            pred = pred[0]  # Extract the tensor if it's part of a tuple
        else:
            pred = pred  # Use directly if it's not a tuple
        
        # Ensure pred is on the correct device and is a tensor
        pred = pred.to(device)
        
        loss_fn = NLLSurvLoss(alpha=0)
        loss = loss_fn(h=pred, y=true_time_bin, t=survtime, c=censor)
        loss_test += loss.data.item()

        # part for nll loss from PORPOISE:
        hazards = torch.sigmoid(pred)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = 1-torch.sum(survival, dim=1).detach().cpu().numpy()
        #risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        all_risk_scores.append(risk)
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(survtime.detach().cpu().numpy())
        all_survival_scores.append(survival.detach().cpu().numpy())
        all_case_ids.append(case_id)

        
        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
    
    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)

    all_risk_scores = np.concatenate(all_risk_scores)
    all_survival_scores = np.concatenate(all_survival_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_case_ids = np.concatenate(all_case_ids)
    
    directory_path = os.path.join(opt.results, 'survival_plots')
    os.makedirs(directory_path, exist_ok=True)
    plot_survival_probabilities(all_survival_scores, all_censorships, all_event_times, all_case_ids, num_cases=10, output_dir=directory_path)

    cindex_test = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    return loss_test, cindex_test
