import os
import logging
import torch
from torch.utils.data import DataLoader
from data_loaders.survival_dataset_5fold import SurvivalDataset  # Adjust import as needed
from train_test.train_test_1modal import train, test, test_and_plot
from options import parse_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Initialize parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)

if not os.path.exists(os.path.join(opt.model_save, opt.exp_name, opt.model_name)):
    os.makedirs(os.path.join(opt.model_save, opt.exp_name, opt.model_name))

# Initialize datasets
data_dir = '/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/data/'
train_data = SurvivalDataset(data_dir, fold=5, seed=41,n_folds=5, split='train')
test_data = SurvivalDataset(data_dir, fold=5, seed=41,n_folds=5, split='test')

train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)

# Train and evaluate the model
model, optimizer, metric_logger = train(opt, train_data, test_data, device)
loss_train, cindex_train = test(opt, model, train_data, device)
loss_test, cindex_test = test_and_plot(opt, model, test_data, device)

# Output results
print(f"[Final] Training set C-Index: {cindex_train:.10f}")
print(f"[Final] Testing set C-Index: {cindex_test:.10f}")

# Save results
if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
    model_state_dict = model.state_dict()
else:
    model_state_dict = model.state_dict()

torch.save({
    'opt': opt,
    'model_state_dict': model_state_dict,
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': metric_logger},
    os.path.join(opt.model_save, opt.exp_name, opt.model_name, 'model.pt')
)

# Optional: Plot and save loss curves
#plt.plot(list(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)), metric_logger['train']['loss'], label='Train Loss')
#plt.plot(list(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)), metric_logger['test']['loss'], label='Test Loss')
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.legend()
#plt.savefig(os.path.join(opt.model_save, opt.exp_name, opt.model_name, 'loss_plot.png'), dpi=300)
#plt.close()

# Save prediction results
#directory_path = os.path.join(opt.results, opt.exp_name, opt.model_name)

# Ensure the directory exists
#os.makedirs(directory_path, exist_ok=True)

# Now safely write the file
#file_path = os.path.join(directory_path, 'pred_train.pkl')
#with open(file_path, 'wb') as f:
   # pickle.dump(pred_train, f)
