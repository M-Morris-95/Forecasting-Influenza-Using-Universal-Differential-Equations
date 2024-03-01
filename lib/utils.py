import os
import logging
import pickle

import torch
import torch.nn as nn
import pandas as pd
import math 
import glob
import re
from shutil import copyfile
import sklearn as sk
import subprocess
import datetime
import numpy as np
import lib.Metrics as Metrics

from filelock import FileLock

def test(model, scaler, x_test, y_test, t, test_season, window_size=1, variables={'ode_name': 'CONN'}, n_samples=128, file_name='results_table'):
    y_pred = model(x_test, t, n_samples=n_samples, training=False)
    y_pr = y_pred.detach().numpy() * scaler.values[np.newaxis, np.newaxis, np.newaxis, :]
    y_te = y_test.detach().numpy() * scaler.values[np.newaxis, np.newaxis, :]

    pred_mean = y_pr.mean(1)
    pred_std = y_pr.std(1)

    lock_path = file_name + ".lock"

    with FileLock(lock_path):
        results_df = pd.read_csv(file_name + '.csv', index_col=0)

        common_indices = None
        for key, value in variables.items():
            try:
                indices = np.where(results_df[key] == value)[0]
                if common_indices is None:
                    common_indices = indices
                else:
                    common_indices = np.intersect1d(common_indices, indices)
            except:
                pass

        if len(common_indices) > 0:
            idx = np.min(common_indices)
        else:
            idx = np.max(results_df.index) + 1

        for key, value in variables.items():
            results_df.loc[idx, key] = value

        for col, g in zip([7,14,21,28], [window_size + 6, window_size + 13, window_size + 20, window_size + 27]):
            results_df.loc[idx, f"{test_season} {g}"] = Metrics.nll(y_te[:, g, :], pred_mean[:, g, :], pred_std[:, g, :])
            results_df.loc[idx, f"skill {test_season} {col}"] = Metrics.skill(y_te[:, g, :], pred_mean[:, g, :], pred_std[:, g, :])

        results_df.to_csv(file_name + '.csv')
               
def append_to_line(file_path, line_prefix, append = 'finished'):
    with FileLock(file_path + ".lock"):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        with open(file_path, 'w') as file:
            for line in lines:
                if line.startswith(line_prefix):
                    line = line.rstrip('\n') + ' '+append+'\n'
                file.write(line)

def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)

def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr

def make_file(prefix):
    root = '/'.join(prefix.split('/')[:-1])
    if not os.path.exists(root):
        os.makedirs(root)
