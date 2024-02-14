import os
import numpy as np
import pandas as pd
import datetime as dt
import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
from scipy.stats import norm
import torch
from itertools import chain
from torch.utils.data import TensorDataset, DataLoader

import lib.Metrics as Metrics
import lib.train_functions as train_functions
import lib.utils as utils

from lib.regional_data_builder import DataConstructor, convert_to_torch
from lib.models import Encoder_Back_GRU, Decoder, Fa, Fp, FaFp
from lib.VAE import VAE

# from lib.encoders import 
from torch import nn
from torchdiffeq import odeint
from torch.distributions import Categorical, Normal
from filelock import FileLock

torch.set_num_threads(1)

if __name__ == '__main__':
    dtype = torch.float32

    # data stuff
    window_size = 28
    gamma = 28
    t = torch.arange(window_size + gamma + 1, dtype=dtype)/7

    # training stuff
    batch_size = 32
    lr = 1e-3
    n_samples = 64
    eval_pts = np.arange(0,t.shape[-1], 7)
    epochs = 250

    # model training info
    training_info =   {
        'UONN': {'nll':True, 'kl_z':True, 'kl_p':True,  'Fa_norm':True,  'reg_loss':True,  'anneal':True},
        'CONN': {'nll':True, 'kl_z':True, 'kl_p':True,  'Fa_norm':False, 'reg_loss':True,  'anneal':True},
        'SONN': {'nll':True, 'kl_z':True, 'kl_p':False, 'Fa_norm':False, 'reg_loss':False, 'anneal':True},
    }

    # model region info 
    region_info = {
        'state': {
            'n_regions': 49,
            'latent_dim': 5,
            'n_qs':5,
            'q_sizes': [256, 128],
            'ff_sizes': [128, 64],
        },
        'hhs': {
            'n_regions': 10,
            'latent_dim': 5,
            'n_qs':9,
            'q_sizes': [128, 64],
            'ff_sizes': [64, 32],
        },
        'US': {
            'n_regions': 1,
            'latent_dim': 6,
            'n_qs':90,
            'q_sizes': [64, 32],
            'ff_sizes': [32, 16],
        }
    }
    ode_names = ['SONN', 'CONN', 'UONN']
    test_seasons = [2015, 2016, 2017, 2018]
    regions = ['US', 'state', 'hhs']
    nums = [1,2,3,4,5]


    started_file_path = "started.txt"

    for num in nums:
        for region in regions:
            for test_season in test_seasons:
                for ode_name in ode_names:
                    

                    # setup files
                    ode = {'CONN':Fp, 'UONN':FaFp, 'SONN':Fa}[ode_name]

                    file_prefix = f'weights/{region}/{ode_name}/{test_season}_{num}_'
                    norm_prefix = f'norms/{region}/{ode_name}/{test_season}_{num}_'
                    chkpt_prefix = f'chkpts/{region}/{ode_name}/{test_season}_{num}_'
                
                    # check if already running
                    lock_path = started_file_path + ".lock"
                    with FileLock(lock_path):
                        run=True
                        with open(started_file_path, 'r') as file:
                            content = file.read().splitlines()
                        
                        for c in content:
                            if file_prefix in c:
                                run = False

                        if run:
                            with open(started_file_path, 'a') as file:
                                file.write(file_prefix + '\n')
                                    
                    if run:            
                        print(region, ode_name, test_season, num)

                        # make folders
                        utils.make_file(chkpt_prefix)
                        utils.make_file(file_prefix)
                        utils.make_file(norm_prefix)

                        # setup stuff for models
                        losses = training_info[ode_name]
                        n_regions = region_info[region]['n_regions']
                        latent_dim = region_info[region]['latent_dim']
                        n_qs = region_info[region]['n_qs']
                        q_sizes = region_info[region]['q_sizes']
                        ff_sizes = region_info[region]['ff_sizes']

                        _data = DataConstructor(test_season=test_season, region = region, window_size=window_size, n_queries=n_qs, gamma=gamma)
                        x_train, y_train, x_test, y_test, scaler = _data(run_backward=True, no_qs_in_output=True)
                        train_loader, x_test, y_test = convert_to_torch(x_train, y_train, x_test, y_test, batch_size=32, shuffle=True, dtype=dtype)  

                        model = VAE(Encoder_Back_GRU, ode, Decoder, n_qs, latent_dim, n_regions, q_sizes=q_sizes, ff_sizes=ff_sizes, file_prefix=file_prefix, chkpt_prefix=chkpt_prefix)
                        model.setup_training(lr=lr)

                        model.pre_train(train_loader, 3, disable=True)
                        model.train(train_loader, t, epochs, losses, eval_pts, n_samples = n_samples, grad_lim=1500, checkpoint=True, track_norms=True, norm_file=f'{norm_prefix}norms.txt', disable=True)
                        model.save()

                        utils.add_finished_to_line(started_file_path, file_prefix)

