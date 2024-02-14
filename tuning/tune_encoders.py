# Standard library imports
import os
import datetime as dt

# Data handling and numerical computations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy import interpolate

# PyTorch related imports
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, kl_divergence
from torch.profiler import profile, record_function, ProfilerActivity
from torchdiffeq import odeint

# Visualization library
import matplotlib.pyplot as plt

# Utilities and custom modules
from itertools import chain
import lib.utils as utils
import lib.models as models
import lib.train_functions as train_functions
import lib.encoders as encoders
from lib.HHS_data import *
import tqdm

# Setting the number of threads for PyTorch and specifying the device
torch.set_num_threads(1)

# root = 'checkpoints/HHS_SIR_Big/'   
# enc.load_state_dict(torch.load(root+'enc_' + '.pth'))
# ode.load_state_dict(torch.load(root+'sir_' + '.pth'))
# dec.load_state_dict(torch.load(root+'dec_' + '.pth'))

def eval(x_in, y_in, t, n_samples = 128, dtype = torch.float32):
    batch_size = x_in.shape[0]
    eps = torch.randn(n_samples, batch_size, n_regions, latent_dim-1, dtype=dtype, device=device)
    ode.clear_tracking()
    mean, std = enc(x_in)
    z = encoders.reparam(eps, std, mean, n_samples, batch_size)
    latent = odeint(ode, z, t, method='rk4', options=dict(step_size = 1.0))
    y_pred = dec(latent[..., :3]).reshape((t.shape[0], n_samples, batch_size, n_regions)).permute(2,1,0,3)

    nll = train_functions.nll_loss(y_pred, y_in).detach().cpu().numpy()
    return nll
    

epochs = 1
def evaluate(params, disable=True, quick=False):
    n_qs = int(params['n_qs'])
    latent_dim = int(params['latent_dim'])
    means = params['means']
    stds = params['stds']
    q_sizes = [int(q) for q in params['q_sizes']]
    ff_sizes = [int(q) for q in params['ff_sizes']]
    ili_sizes = [int(q) for q in params['ili_sizes']]
    SIR_scaler = params['SIR_scaler']
    ls_enc_name = params['ls_enc_name']

    if enc_name == 'Encoder_BiDirectionalGRU':
        encoder_model = encoders.Encoder_BiDirectionalGRU
    if enc_name == 'Encoder_MISO_GRU':
        encoder_model = encoders.Encoder_MISO_GRU
    if enc_name == 'Encoder_Back_GRU':
        encoder_model = encoders.Encoder_Back_GRU


    suppress_outputs = True
    lag = 14
    n_regions = 10
    season = 2016
    lr = 1e-3
    n_samples = 128
    
    
    root = 'checkpoints/HHS_SIR_Big_new/'      
    device = 'cpu'
    dtype=torch.float32
    
    tmax = 8
    
    if encoder_model == encoders.Encoder_Back_GRU:   
        gamma = 28
        t = torch.linspace(1,gamma+window, gamma+window, device=device)/7
        
    else:
        gamma = 63
        t = torch.linspace(1,gamma, gamma, device=device)/7
    eval_pts = [0,6,13,20,27,34,40,47,54][:tmax]
    
    ili = load_ili('hhs')
    ili = intepolate_ili(ili)
    
    hhs_dict = {}
    qs_dict = {}
    
    ignore = ['AZ', 'ND', 'AL', 'RI', 'VI', 'PR']
    for i in range(1,1+n_regions):
        hhs_dict[i] = get_hhs_query_data(i, ignore=ignore, smooth_after = True)
        qs_dict[i] = choose_qs(hhs_dict, ili, i, season, n_qs)
    
        hhs_dict[i] = hhs_dict[i].loc[:, list(qs_dict[i])]
        hhs_dict[i] = hhs_dict[i].div(hhs_dict[i].max())
        
    ili = ili.loc[hhs_dict[i].index[0] : hhs_dict[i].index[-1]]
    ili = ili.div(ili.max())
    
    run_backward = False
    if encoder_model == encoders.Encoder_Back_GRU:
        run_backward = True
    
    
    # Build inputs
    inputs = []
    outputs = []
    for batch in range(ili.shape[0] - (window+gamma)):
        batch_inputs = []
        for i in range(1,11):
            batch_inputs.append(hhs_dict[i].iloc[batch:batch+window])
        
        t_ili = ili.iloc[batch:batch+window].copy()
        t_ili.iloc[-lag:, :] = -1
        batch_inputs.append(t_ili)
        batch_inputs = np.concatenate(batch_inputs, -1)
    
        if run_backward:
            gamma = 28
            batch_outputs = ili.iloc[batch:batch+window-lag+gamma].values
            t = torch.linspace(1, batch_outputs.shape[0], batch_outputs.shape[0])/7
        else:
            gamma = 56
            batch_outputs = ili.iloc[batch+window-lag:batch+window-lag+gamma].values
            t = torch.linspace(1, batch_outputs.shape[0], batch_outputs.shape[0])/7
            
        inputs.append(batch_inputs)
        outputs.append(batch_outputs)
    inputs = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    outputs = torch.tensor(np.asarray(outputs), dtype=torch.float32)
    
    
    # Load models
    enc = encoder_model(n_regions, 
                 n_qs=n_qs,
                 latent_dim = latent_dim-1,    
                 q_sizes=q_sizes, 
                 ili_sizes=ili_sizes, 
                 ff_sizes = ff_sizes, 
                 SIR_scaler = SIR_scaler, 
                 device=device, 
                 dtype=torch.float32)
    
    ode = models.Fp(n_regions, latent_dim, nhidden=64)
    dec = models.Decoder(n_regions, 3, 1, device=device)
    
    enc.to(device)
    ode.to(device)
    dec.to(device)
    
    if not suppress_outputs:
        num = np.sum([np.prod(_.shape) for _ in list(enc.parameters())])
        print('encoder parameters:', num)
        
        num = np.sum([np.prod(_.shape) for _ in list(ode.parameters())])
        print('ode parameters:', num)
        
        num = np.sum([np.prod(_.shape) for _ in list(dec.parameters())])
        print('decoder parameters:', num)
    
    batch_size = 32
    new_inputs = torch.tensor(np.asarray(inputs), dtype=torch.float32).to(device)
    new_outputs = torch.tensor(np.asarray(outputs), dtype=torch.float32).to(device)
    
    train_size = len(new_inputs) - 365
    x_tr, y_tr = new_inputs[:train_size], new_outputs[:train_size]
    x_test, y_test = new_inputs[train_size:], new_outputs[train_size:]
    
    # batch it all 
    x_train = []
    y_train = []
    for b in range(int(np.ceil(x_tr.shape[0]/batch_size))):
        x_train.append(torch.tensor(x_tr[b*batch_size:(b+1)*batch_size], dtype=torch.float32))
        y_train.append(torch.tensor(y_tr[b*batch_size:(b+1)*batch_size], dtype=torch.float32))
    
    # pre train
    optimizer = torch.optim.Adam(enc.parameters(), lr=lr)
    for epoch in range(3):
        kls = 0
        pbar = tqdm.tqdm(x_train, disable=suppress_outputs)
        num = 0
        for x_tr in pbar:
            optimizer.zero_grad()
            
            mean, std = enc(x_tr)
            prior = encoders.make_prior(mean, latent_dim=latent_dim, device=device)
            kl = kl_divergence(Normal(mean, std), prior).mean(0).sum()
            if torch.isnan(kl):
                break
            kl.backward()
            optimizer.step()
            kls += kl.cpu().detach().numpy()
            num += 1
            pbar.set_postfix({'Epoch':epoch, 'KL_z':kls/num})
    
    
    optimizer = torch.optim.Adam(chain(enc.parameters(), ode.parameters(), dec.parameters()), lr=lr)
    _history = train_functions.history()
    
    for epoch in range(epochs):
        pbar = tqdm.tqdm(zip(x_train, y_train), disable=suppress_outputs)
        for x_tr, y_tr in pbar:
            batch_size = x_tr.shape[0]
            eps = torch.randn(n_samples, batch_size, n_regions, latent_dim-1, dtype=dtype, device=device)
            ode.clear_tracking()
            optimizer.zero_grad()
            
            mean, std = enc(x_tr)
            z = encoders.reparam(eps, std, mean, n_samples, batch_size)
            latent = odeint(ode, z, t, method='rk4', options=dict(step_size = 1.0))
            y_pred = dec(latent[..., :3]).reshape((-1, n_samples, batch_size, n_regions)).permute(2,1,0,3)
    
            # nll = train_functions.nll_loss(y_pred, y_tr[:, eval_pts, :])
            nll = train_functions.nll_loss(y_pred, y_tr)
            kl_p = train_functions.get_kl_params(1, ode.posterior(), means=means, stds = stds,limit = 1e6, device=device)
            kl_z = kl_divergence(encoders.make_prior(mean, latent_dim=latent_dim, device=device), Normal(mean, std)).sum(-1).mean() / len(x_train)
            reg_loss = train_functions.latent_init_loss(latent[..., :3])
    
            loss = nll+kl_p+kl_z+reg_loss
            loss.backward()
            optimizer.step()
            _history.batch([loss.cpu(), nll.cpu(), kl_z.cpu(),kl_p.cpu(),reg_loss.cpu(), optimizer.param_groups[-1]['lr']], ['loss', 'nll', 'kl_latent', 'kl_params', 'reg_loss', 'lr'])
            pbar.set_postfix(_history.epoch())
        _history.reset()
            
        utils.update_learning_rate(optimizer, 0.999, lr/10)
    
    val_nll = eval(x_test, y_test, t, n_samples = 128, dtype = torch.float32)
    score = train_functions.nll_loss(y_pred, y_test.unsqueeze(-2)).detach().numpy()

    return score





if __name__ == "__main__":
    import time
    import random
    from filelock import FileLock
    import sys
    import time
    import argparse

    position = int(sys.argv[1])
    lock = FileLock("validation_scores.csv.lock")
    for _ in range(256):
        with lock:
            df = pd.read_csv('validation_scores.csv', index_col=0)
            try:
                param_num = np.min(np.where(df['started'] == 0))
                print(_, param_num)
            except:
                print('oh no')
            df.loc[param_num, 'started'] = 1
            df.to_csv('validation_scores.csv')

        score = 10
        try:
            print('starting:', df.loc[param_num])
            score = evaluate(dict(df.loc[param_num]), disable=True)
        except Exception as e:
            print("An error occurred:", e)


        with lock:
            df = pd.read_csv('validation_scores.csv', index_col=0)
            df.loc[param_num, 'score'] = score
            df.to_csv('validation_scores.csv')





