import os
import logging
import torch
from torch.utils.data import DataLoader
from survival_dataset_5modal import SurvivalDataset  # Adjust import as needed
from train_test_1modal import train, test, test_and_plot
from options import parse_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import statistics

# Initialize parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)

if not os.path.exists(os.path.join(opt.model_save, opt.exp_name, opt.model_name)):
    os.makedirs(os.path.join(opt.model_save, opt.exp_name, opt.model_name))

# Initialize datasets
data_dir = '/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/data/'

# Train and evaluate the model

c_indices_per_seed = []
for seed in range(1,11):

    c_indices_per_fold = []
    for fold in range(1, 6):

        train_data = SurvivalDataset(data_dir, fold, seed,5, split='train')
        test_data = SurvivalDataset(data_dir, fold, seed,5, split='test')
        model, optimizer, metric_logger = train(opt, train_data, test_data, device)
        loss_test, cindex_test = test(opt, model, test_data, device)
        c_indices_per_fold.append(cindex_test)

    average_c_index = sum(c_indices_per_fold) / len(c_indices_per_fold)
    print("Average C-Index(seed {seed}):", average_c_index)
    c_indices_per_seed.append(average_c_index)
average_c_index_final = sum(c_indices_per_seed) / len(c_indices_per_seed)
std_dev_c_index_final = statistics.stdev(c_indices_per_seed)
# Output results
print(f"[Final] Average C-Index across 10 5-fold tests: {average_c_index_final:.10f}")
print(f"[Final] Standard deviation across 10 5-fold tests: {std_dev_c_index_final:.10f}")

