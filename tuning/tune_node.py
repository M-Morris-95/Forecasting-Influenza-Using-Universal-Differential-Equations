import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from itertools import chain
import tqdm
import torch
import copy
from torch import nn
from torch.distributions import Normal, Categorical, kl_divergence
from torchdiffeq import odeint

# Set device and torch threads
device = 'cpu'
torch.set_num_threads(1)

# Custom library imports
import lib.utils as utils
import lib.Data_Constructor as Data_Constructor
import lib.train_functions as train_functions
import lib.models as models
import lib.osthus_stuff as osthus_stuff

class Encoder_BiDirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim, SIR_scaler = [0.1, 0.05, 1.0], device='cpu', dtype=torch.float32):
        super(Encoder_BiDirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.scaler = torch.tensor(SIR_scaler, dtype=dtype, device=device)
        if latent_dim > len(self.scaler):
            extension = self.scaler[-1].repeat(latent_dim - len(self.scaler))
            self.scaler = torch.cat([self.scaler, extension])
        self.scaler = self.scaler.view(1, -1)
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected output layer
        self.latent_dim=latent_dim
        self.fc = nn.Linear(hidden_size * 2, 2*latent_dim)  # Multiply hidden_size by 2 for bidirectional

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -14, :])

        mean, std = utils.split_last_dim(out)

        mean = mean.reshape(-1, self.latent_dim)
        std = std.reshape(-1, self.latent_dim)
        
        std = torch.abs(std) * self.scaler
        return mean.unsqueeze(-2), std.unsqueeze(-2)

class Fp(nn.Module):
    def __init__(self, n_region=1, latent_dim=8, nhidden=20):
        super(Fp, self).__init__()

        self.n_region = n_region
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_region*latent_dim, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, 2*n_region),
        )
        
        self.params = []

    def forward(self, t, x):
        out_of_range_mask = (x > 2) | (x < -1)
        out = torch.abs(self.net(x)).reshape(-1, self.n_region, 2)
        
        self.params.append(out)

        plusI = out[..., 0] * x[..., 0] * x[..., 1]
        minusI = out[..., 1] * x[..., 1]

        Fp = torch.stack([-plusI, plusI - minusI, minusI], dim=-1)
        
        res = torch.cat([Fp, torch.zeros_like(x[..., 3:])], -1)
        res[out_of_range_mask] = 0.0
        return res
    
    def clear_tracking(self):
        """Resets the trackers."""
        self.params = []
    
    def posterior(self):
        params = torch.stack(self.params).reshape(-1, 2)
        
        self.params = []
        return Normal(params.mean(0), params.std(0))

def make_prior(mean, z_prior=torch.tensor([0.1, 0.01]), device='cpu', latent_dim=8):
    mean_concat = torch.cat((mean[..., :2]  , torch.zeros_like(mean[..., 2:], device=device)), dim=-1)
    std = torch.cat([z_prior[0].unsqueeze(0), z_prior[1].unsqueeze(0), torch.ones(latent_dim - len(z_prior) - 1, device=device)], 0).expand_as(mean_concat)

    return Normal(mean_concat, torch.abs(std))

def reparam(eps, std, mean, n_samples, batch_size):
    z = eps * std + mean
    z = torch.concat([torch.abs(z[..., :2]), (1 - torch.abs(z[..., :2]).sum(-1)).unsqueeze(-1), z[..., 2:]], -1)
    z = z.reshape((n_samples * batch_size, ) + z.shape[2:])
    return z
    





def evaluate(params, disable=True, quick=False):
    input_size = int(params['input_size'])
    hidden_size = int(params['hidden_size'])
    num_layers = int(params['num_layers'])
    latent_dim = int(params['latent_dim'])
    n_samples = int(params['n_samples'])
    window_size = int(params['window_size'])
    batch_size = int(params['batch_size'])
    # root = params['root']
    epochs = int(params['epochs'])
    
    test_season = 2016
    data_season = 2015
    means=[0.8, 0.55]
    stds = [0.2, 0.2]
    lr = 1e-3
    device='cpu'
    gamma = 56
    tmax = 56
       


    _data = Data_Constructor.DataConstructor(test_season, data_season, gamma, window_size, n_queries=input_size-1, selection_method='distance')
    _data()
    inputs, outputs, dates, test_dates = _data.build()
    
    test_start = np.where(dates == test_dates[0])[0][0]
    dtype = torch.float32
    x_tr = torch.tensor(inputs[:test_start], dtype=dtype)
    x_test = torch.tensor(inputs[test_start:], dtype=dtype)
    y_tr = torch.tensor(outputs[:test_start], dtype=dtype)
    y_test = torch.tensor(outputs[test_start:], dtype=dtype)
    
    n_batches = int(np.ceil(x_tr.shape[0]/batch_size))
    # batch it all 
    x_train = []
    y_train = []

    for b in range(n_batches):
        x_batch = x_tr[b*batch_size:(b+1)*batch_size].clone().detach()
        y_batch = y_tr[b*batch_size:(b+1)*batch_size].clone().detach().unsqueeze(-2)
        x_train.append(x_batch)
        y_train.append(y_batch)

    # Create the model
    n_regions = 1
    enc = Encoder_BiDirectionalLSTM(input_size, hidden_size, num_layers, latent_dim-1)
    ode = Fp(n_regions, latent_dim, nhidden=32)
    dec = models.Decoder(n_regions, 3, 1)
    
    optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)
    for epoch in range(30):
        kls = 0
        for x_tr in x_train:
            optimizer.zero_grad()
            
            mean, std = enc(x_tr)
            prior = make_prior(mean, latent_dim=latent_dim)
            kl = kl_divergence(Normal(mean, std), prior).mean(0).sum()
            kl.backward()
            optimizer.step()
            kls += kl.detach().numpy()
            
    
    optimizer = torch.optim.Adam(chain(enc.parameters(), ode.parameters(), dec.parameters()), lr=1e-3)
    _history = train_functions.history()
    
    for epoch in range(epochs):
        
        t = torch.linspace(1, tmax, tmax)/7
        epc = len(_history.epoch_history)
        for x_tr, y_tr in zip(x_train, y_train):
            batch_size = x_tr.shape[0]
            eps = torch.tensor(np.random.normal(0, 1,  (n_samples, batch_size, n_regions, latent_dim-1)), dtype=dtype, device=device)
            ode.clear_tracking()
            
            optimizer.zero_grad()
    
            
            mean, std = enc(x_tr)
            z = reparam(eps, std, mean, n_samples, batch_size)
            latent = odeint(ode, z, t, method='rk4', options=dict(step_size = 1.0))
            y_pred = dec(latent[..., :3]).reshape((tmax, n_samples, batch_size, n_regions)).permute(2,1,0,3)
            
            nll = train_functions.nll_loss(y_pred, y_tr[:, :t.shape[0], :])
            kl_p = train_functions.get_kl_params(1, ode.posterior(), means=means, stds = stds,limit = 500)
            kl_z = kl_divergence(make_prior(mean, latent_dim=latent_dim), Normal(mean, std)).sum(-1).mean() / len(x_train)
            reg_loss = train_functions.latent_init_loss(latent[..., :3])
    
            loss = nll+kl_p+kl_z+reg_loss
            loss.backward()
            optimizer.step()
            _history.batch([loss.cpu(), nll.cpu(), kl_z.cpu(),kl_p.cpu(),reg_loss.cpu(), optimizer.param_groups[-1]['lr'], tmax], ['loss', 'nll', 'kl_latent', 'kl_params', 'reg_loss', 'lr', 'tmax'])
        _history.reset()
        if epoch > 10:
            if np.all([h['nll'] < -2 for h in _history.epoch_history[-10:]]):
                tmax = min(tmax+1, 28)
        
        utils.update_learning_rate(optimizer, 0.999, lr/10)
        
    mean, std = enc(x_test)
    batch_size = x_test.shape[0]
    eps = torch.tensor(np.random.normal(0, 1,  (n_samples, batch_size, n_regions, latent_dim-1)), dtype=dtype, device=device)
    z = reparam(eps, std, mean, n_samples, batch_size)
    latent = odeint(ode, z, t, method='rk4', options=dict(step_size = 1.0))
    y_pred = dec(latent[..., :3]).reshape((tmax, n_samples, batch_size, n_regions)).permute(2,1,0,3)*7.7151
    
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