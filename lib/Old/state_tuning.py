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
from lib.state_data import *
import tqdm
import ast

# Setting the number of threads for PyTorch and specifying the device
torch.set_num_threads(1)

def test(x_in, y_in, t, enc, ode, dec, n_samples = 512, n_regions=10, latent_dim=6, dtype = torch.float32, device='cpu', suppress_outputs = False):
    with torch.no_grad():
        ideal_batch_size = 4
        
        t = torch.arange(y_in.shape[1])/7
        n_batches = int(np.ceil(x_in.shape[0]/ideal_batch_size))

        y_means = []
        y_stds = []
        for batch in tqdm.trange(n_batches, disable=suppress_outputs):
            x_batch = x_in[batch * ideal_batch_size:(batch+1)*ideal_batch_size]
            batch_size = x_batch.shape[0]
            eps = torch.randn(n_samples, batch_size, n_regions, latent_dim-1, dtype=dtype, device=device)
            ode.clear_tracking()
            mean, std = enc(x_batch)
            z = encoders.reparam(eps, std, mean, n_samples, batch_size)
            latent = odeint(ode, z, t, method='rk4', options=dict(step_size = 1.0))

            y_pred = dec(latent[..., :3]).reshape((t.shape[0], n_samples, batch_size, n_regions)).permute(2, 1, 0, 3)

            y_means.append(y_pred.mean(1))
            y_stds.append(y_pred.std(1))
            
        y_means = torch.concat(y_means, 0)
        y_stds = torch.concat(y_stds, 0)
        nll = -Normal(y_means, y_stds).log_prob(y_in).mean()
        return nll
    
def evaluate(params, encoder_model = encoders.Encoder_Back_GRU,  suppress_outputs = True, dtype = torch.float32, device = 'cpu'):
    n_qs = int(params['n_qs'])
    latent_dim = int(params['latent_dim'])
    means = ast.literal_eval(params['means'])
    stds = ast.literal_eval(params['stds'])
    q_sizes = ast.literal_eval(params['q_sizes'])
    ff_sizes = ast.literal_eval(params['ff_sizes'])
    SIR_scaler = ast.literal_eval(params['SIR_scaler'])
    anneal = int(params['anneal'])
    epochs = int(params['epochs'])
    test_n_samples = int(params['test_n_samples'])
    season = int(params['season'])
    
    lr = 1e-3
    lag = 14
    n_regions = 49
    window = 35
    lag =14
    gamma = 28
    batch_size=32

    print('loading data')
    x_train, y_train, x_test, y_test = build_data(n_qs, season, window = window, gamma = gamma, lag = lag, batch_size=batch_size, validation=True, region = 'state', ignore = ['FL'], root = '../google_queries/', append = 'state_queries_new')

    print('setting up')
    enc = encoder_model(n_regions, 
                    n_qs=n_qs,
                    latent_dim = latent_dim-1,    
                    q_sizes=q_sizes, 
                    ili_sizes=None, 
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

    # pre train
    print('pre training')
    optimizer = torch.optim.Adam(enc.parameters(), lr=lr)
    for epoch in range(10):
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

    kl_w = 1
    step = 0
    n_samples_dict = {100:32, 250:64, 400:128}

    optimizer = torch.optim.Adam(chain(enc.parameters(), ode.parameters(), dec.parameters()), lr=lr)
    _history = train_functions.history()

    t = torch.arange(y_test.shape[1])/7
    eval_pts = np.arange(0,t.shape[0], 7)

    track_norms = []
    print('training')

    for epoch in range(epochs):
        for key in sorted(n_samples_dict.keys(), reverse=False):
            if epoch <= key:
                n_samples = n_samples_dict[key]
                break

        batch_grad_norms = []
        pbar = tqdm.tqdm(zip(x_train, y_train), disable=suppress_outputs)
        for x_tr, y_tr in pbar:
            batch_size = x_tr.shape[0]
            if anneal:
                step += 1 
                kl_w = train_functions.KL_annealing(step, reset_pos=10000, split=0.5, lower = 0.0, upper = 1.0, type = 'cosine')
            eps = torch.randn(n_samples, batch_size, n_regions, latent_dim-1, dtype=dtype, device=device)
            ode.clear_tracking()
            optimizer.zero_grad()

            mean, std = enc(x_tr)
            z = encoders.reparam(eps, std, mean, n_samples, batch_size)
            latent = odeint(ode, z, t[eval_pts], method='rk4', options=dict(step_size = 1.0))
            y_pred = dec(latent[..., :3]).reshape((-1, n_samples, batch_size, n_regions)).permute(2,1,0,3)

            nll = train_functions.nll_loss(y_pred, y_tr[:, eval_pts, :])
            kl_p = train_functions.get_kl_params(1, ode.posterior(), means=means, stds = stds,limit = 1e6, device=device)
            kl_z = kl_w*kl_divergence(encoders.make_prior(mean, latent_dim=latent_dim, device=device), Normal(mean, std)).sum(-1).mean() / len(x_train)
            reg_loss = train_functions.latent_init_loss(latent[..., :3])

            loss = nll+kl_p+kl_z+reg_loss
            loss.backward()


            # Check gradient magnitudes
            grad_norm = torch.norm(torch.cat([p.grad.data.view(-1) for p in chain(enc.parameters(), ode.parameters(), dec.parameters())]), 2).item()
            batch_grad_norms.append(grad_norm)

            gradient_threshold = 300
            if grad_norm < gradient_threshold or epoch <= 3:
                optimizer.step()

            # _history.batch([loss.cpu(), nll.cpu(), kl_z.cpu(),kl_p.cpu(),reg_loss.cpu(), optimizer.param_groups[-1]['lr'], kl_w], ['loss', 'nll', 'kl_latent', 'kl_params', 'reg_loss', 'lr', 'kl_w'])
            _history.batch([round(loss.cpu().item(), 3), 
                            round(nll.cpu().item(), 3), 
                            round(kl_z.cpu().item(), 3), 
                            round(kl_p.cpu().item(), 3), 
                            round(reg_loss.cpu().item(), 3), 
                            round(optimizer.param_groups[-1]['lr'], 3), 
                            round(kl_w, 3), n_samples], 
                            ['loss', 'nll', 'kl_latent', 'kl_params', 'reg_loss', 'lr', 'kl_w', 'n_samples'])

            pbar.set_postfix(_history.epoch())
        _history.reset()

        if not suppress_outputs:
            with open('grad_norms.txt', 'a') as file:
                file.write(','.join(map(str, batch_grad_norms)) + '\n')

        rounded_epoch_history = {key: round(value, 3) for key, value in _history.epoch_history[-1].items()}
        rounded_epoch_history['grad_norm'] = max(batch_grad_norms)
        if suppress_outputs:
            print(epoch + 1, dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rounded_epoch_history)
        utils.update_learning_rate(optimizer, 0.999, lr/10)

    nll = test(x_test, y_test, t, enc, ode, dec, n_samples = test_n_samples, n_regions=n_regions, latent_dim=latent_dim, dtype = torch.float32, device='cpu', suppress_outputs = suppress_outputs)

    return nll.item()

if __name__ == "__main__":
    import time
    import datetime as dt
    import random
    from filelock import FileLock
    import sys
    import time
    import argparse
    import traceback

    try:
       
        position = int(sys.argv[1])
        suppress_outputs = True
        do_started_thing = True
        print("running from script")
    except:
        print('running not from bash')
        suppress_outputs = False
        do_started_thing = False

    lock = FileLock("state_validation_scores.csv.lock")
    for _ in range(256):
        print('before lock')
        with lock:
            print('with lock')
            df = pd.read_csv('state_validation_scores.csv', index_col=0)
            try:
                if isinstance(df['started'][0], str):
                    param_num = np.min(np.where(df['started'] == '0'))
                else:
                    param_num = np.min(np.where(df['started'] == 0))

                print('run:', _, 'param num:', param_num)
            except:
                print('oh no')
            if do_started_thing:

                current_time = dt.datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                df.loc[param_num, 'started'] = formatted_time
                df.to_csv('state_validation_scores.csv')

        score = 10
        try:
            print('starting:', df.loc[param_num])
            score = evaluate(dict(df.loc[param_num]), suppress_outputs=suppress_outputs)
            print(score)
        except Exception as e:
            traceback.print_exc()
            print("An error occurred:", e)


        with lock:
            if do_started_thing:
                df = pd.read_csv('state_validation_scores.csv', index_col=0)
                df.loc[param_num, 'score'] = score
                df.to_csv('state_validation_scores.csv')
