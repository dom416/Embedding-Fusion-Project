import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from HFB_fusion import HFBSurv
from torch.utils.data import DataLoader
from survival_dataset import SurvivalDataset
from utils import CoxLoss, regularize_weights, CIndex_lifeline, cox_log_rank, accuracy_cox,count_parameters
import torch.optim as optim
import pickle
import os
import gc

####################################################### MFB ############################################################

def train(opt,train_data,test_data,device,k):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)
    model = HFBSurv((1024, 1024, 1024), (48, 48, 48, 256), (20, 20, 1), (0.2, 0.2, 0.2, 0.3), 20, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=True)
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    c_index_best = 0

    for epoch in tqdm(range(opt.epoch_count, opt.niter+opt.niter_decay+1)):
        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        loss_epoch = 0
        gc.collect()
        for batch_idx, batch in enumerate(train_loader):
            censor = batch['censor'].to(device)
            survtime = batch['survival_time'].to(device)
            x_clin = batch['clinical_data'].to(device)
            x_pathr = batch['pathology_report'].to(device)
            x_unis = batch['uni_slide_image'].to(device)
          
            pred, _ = model(x_clin, x_pathr, x_unis)
            #print(f"Batch {batch_idx}: Predictions - {pred}")
            
            loss_cox = CoxLoss(survtime, censor, pred, device)
            loss_reg = regularize_weights(model=model)
            loss = loss_cox + opt.lambda_reg *  loss_reg


            loss_epoch += loss_cox.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_epoch /= len(train_loader.dataset)

            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
            pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
            surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)
            loss_test, cindex_test, pvalue_test, surv_acc_test,pred_test,code_test = test(opt, model, test_data, device)

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
            metric_logger['train']['pvalue'].append(pvalue_epoch)
            metric_logger['train']['surv_acc'].append(surv_acc_epoch)

            metric_logger['test']['loss'].append(loss_test)
            metric_logger['test']['cindex'].append(cindex_test)
            metric_logger['test']['pvalue'].append(pvalue_test)
            metric_logger['test']['surv_acc'].append(surv_acc_test)

            if cindex_test > c_index_best:
                c_index_best = cindex_test
            if opt.verbose > 0:
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))

    return model, optimizer, metric_logger

def test(opt,model, test_data, device):
    model.eval()
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]),np.array([])
    loss_test = 0
    code_final = None

    for batch_idx, batch in enumerate(test_loader):
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        x_clin = batch['clinical_data'].to(device)
        x_pathr = batch['pathology_report'].to(device)
        x_unis = batch['uni_slide_image'].to(device)
        
        pred, code = model(x_clin, x_pathr, x_unis)
        loss_cox = CoxLoss(survtime, censor, pred, device)
        loss_test += loss_cox.data.item()
        
        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
    
    
        if batch_idx ==0:
            code_final = code
        else:
            code_final = torch.cat([code_final,code])
    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)
    cindex_test =  CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    pred_test = [risk_pred_all, survtime_all, censor_all]
    code_final_data = code_final.data.cpu().numpy()
    return loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test, code_final_data