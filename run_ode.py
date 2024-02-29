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
from lib.in_development.models_bayes import Bayes_Fa, Bayes_Fp, Bayes_FaFp
from lib.VAE import VAE

from torch import nn
from torchdiffeq import odeint
from torch.distributions import Categorical, Normal
from filelock import FileLock

torch.set_num_threads(1)

dtype = torch.float32

# data stuff
window_size = 1

# training stuff
batch_size = 32
lr = 1e-3
n_samples = 64

# model region info 
region_info = {
    'state': {
        'n_regions': 49,
        'latent_dim': 8,
        'n_qs':8,
        'ode_params':{'net_sizes': [64, 64, 32],  'aug_net_sizes': [64, 64], 'prior_std' : 0.05},
        'dec_params':{},
        'enc_params':{'q_sizes':[256, 128], 'ff_sizes':[64,64], 'SIR_scaler':[0.1, 0.05, 1.0]},
        'epochs':120
    },
    'hhs': {
        'n_regions': 10,
        'latent_dim': 8,
        'n_qs':15,
        'ode_params':{'net_sizes': [64, 64, 32],  'aug_net_sizes': [64, 64], 'prior_std' : 0.05},
        'dec_params':{},
        'enc_params':{'q_sizes':[256, 128], 'ff_sizes':[64,64], 'SIR_scaler':[0.1, 0.05, 1.0]},
        'epochs':120
    },
    'US': {
        'n_regions': 1,
        'latent_dim': 8,
        'n_qs':90,
        'ode_params':{'net_sizes': [64, 64, 32],  'aug_net_sizes': [64, 64], 'prior_std' : 0.05},
        'dec_params':{},
        'enc_params':{'q_sizes':[256, 128], 'ff_sizes':[64,64], 'SIR_scaler':[0.1, 0.05, 1.0]},
        'epochs':120
    }
}

# model training info
training_info =   {
    'UONN': {'nll':True, 'mse':False, 'kl_z':True, 'kl_p':True,  'Fa_norm':1e-2,  'reg_loss':True,  'anneal':True},
    'CONN': {'nll':True, 'mse':False, 'kl_z':True, 'kl_p':True,  'Fa_norm':False, 'reg_loss':True,  'anneal':True},
    'SONN': {'nll':True, 'mse':False, 'kl_z':True, 'kl_p':False, 'Fa_norm':False, 'reg_loss':False, 'anneal':True},
    'UONNb': {'nll':True, 'mse':False, 'kl_z':True, 'kl_p':True,  'Fa_norm':True,  'reg_loss':True,  'anneal':True},
    'CONNb': {'nll':True, 'mse':False, 'kl_z':True, 'kl_p':True,  'Fa_norm':False, 'reg_loss':True,  'anneal':True},
    'SONNb': {'nll':True, 'mse':False, 'kl_z':True, 'kl_p':False, 'Fa_norm':False, 'reg_loss':False, 'anneal':True},
}

ode_names = ['CONN', 'UONN']
test_seasons = [2015,2016,2017,2018]
regions = ['US', 'hhs', 'state']
epoch_ls = [10,140,200,260]
nums = [15,16,17,18,19]
uncertainty=True

started_file_path = "started.txt"
for epochs in epoch_ls:
    for gamma in [35,42,49,56]:
        for latent_dim in [8]:
            for num in nums:
                for region in regions:
                    for test_season in test_seasons:
                        for ode_name in ode_names:
                            # setup files
                            ode = {'CONN':Fp, 'UONN':FaFp, 'SONN':Fa, 'CONNb':Bayes_Fp, 'UONNb':Bayes_FaFp, 'SONNb':Bayes_Fa}[ode_name]

                            common_prefix = f'{region}/{ode_name}/{test_season}_e{epochs}_g{gamma}_{num}_'
                            file_prefix = f'weights/{common_prefix}'
                            norm_prefix = f'norms/{common_prefix}'
                            chkpt_prefix = f'chkpts/{common_prefix}'

                            # check if already running
                            lock_path = started_file_path + ".lock"
                            with FileLock(lock_path):
                                run=True
                                with open(started_file_path, 'r') as file:
                                    content = file.read().splitlines()
                                
                                for c in content:
                                    a=1
                                    if file_prefix in c:
                                        run = False
                                if run:
                                    with open(started_file_path, 'a') as file:
                                        file.write(file_prefix + '\n')
                            if run:      
                                try:      
                                    print(region, ode_name, test_season, epochs, num)

                                    # make folders
                                    utils.make_file(chkpt_prefix)
                                    utils.make_file(file_prefix)
                                    utils.make_file(norm_prefix)
                                
                                    # setup stuff for models
                                    t = torch.arange(window_size + gamma + 1, dtype=dtype)/7
                                    eval_pts = np.arange(0,t.shape[-1], 7)

                                    losses = training_info[ode_name]
                                    n_regions = region_info[region]['n_regions']
                                    n_qs = region_info[region]['n_qs']
                                    enc_params = region_info[region]['enc_params']
                                    dec_params = region_info[region]['dec_params']
                                    ode_params = region_info[region]['ode_params']

                                    _data = DataConstructor(test_season=test_season, region = region, window_size=window_size, n_queries=n_qs, gamma=gamma)
                                    x_train, y_train, x_test, y_test, scaler = _data(run_backward=True, no_qs_in_output=True)
                                    train_loader, x_test, y_test = convert_to_torch(x_train, y_train, x_test, y_test, batch_size=32, shuffle=True, dtype=dtype)  

                                    model = VAE(Encoder_Back_GRU, ode, Decoder, n_qs, latent_dim, n_regions, file_prefix=file_prefix, chkpt_prefix=chkpt_prefix, ode_params=ode_params, enc_params=enc_params, dec_params=dec_params, uncertainty=uncertainty, ode_kl_w = 1/153)
                                    model.setup_training(lr=lr)

                                    eval_all = list(np.linspace(0, gamma, int((gamma/7)+1), dtype=int))
                                    epochs_per_cycle = int(epochs/(len(eval_all)-1))
                                    for i in range(2,len(eval_all)+1):
                                        eval_pts = eval_all[:i]
                                        time_steps = t[:(eval_pts[-1]+1)]
                                        
                                        model.train(train_loader, 
                                                    time_steps, 
                                                    epochs_per_cycle, 
                                                    losses, 
                                                    eval_pts, 
                                                    n_samples = n_samples, 
                                                    grad_lim=5000, 
                                                    checkpoint=True, 
                                                    track_norms=True, 
                                                    norm_file=f'{norm_prefix}norms.txt', 
                                                    disable=True, 
                                                    validate = {'x_test':x_test, 'y_test':y_test, 't':t, 'scaler':scaler, 'n_samples':32})            
                                    model.save()
                                    
                                    utils.test(model, scaler, x_test, y_test, t, test_season=test_season, variables = {'epochs':epochs, 'gamma':gamma, 'ode_name':ode_name, 'region':region, 'latent_dim':latent_dim, 'num':num}, n_samples = 128, file_name='results_table_server')
                                    utils.append_to_line(started_file_path, file_prefix, append = 'finished')
                                except:
                                    utils.append_to_line(started_file_path, file_prefix, append = 'failed')
                                    pass
                        
