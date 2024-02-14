import pandas as pd
import os
import torch
from torch import nn
from itertools import chain
from torch.optim.lr_scheduler import CyclicLR
from torch.distributions import Normal
import numpy as np
from torchdiffeq import odeint
import tqdm
import matplotlib.pyplot as plt
torch.set_num_threads(1)

from lib.data import data
import lib.models as models
import lib.utils as utils
import lib.train_functions as train_functions

def evaluate(params, disable=True, quick=False):
    latent_dim = int(params['latent_dim'])
    rnn_hidden_size = int(params['rnn_hidden_size'])
    limit = int(params['limit'])
    epochs = int(params['epochs'])
    nhidden = int(params['nhidden'])
    lr = params['lr']
    lr_scale = params['lr_scale']
    klp_w = params['klp_w']
    klz_w = params['klz_w']
    Sstddev = params['Sstddev']
    Istddev = params['Istddev']
    
    # preset
    batch_size = 16
    input_dim = 1
    dtype=torch.float32
#     country = 'England'
    country ='US'
    n_samples = 128
    window = 5
    gamma = 5
    device = "cpu"
    disable = True
    year = 2015
    
    
    x_train, y_train, x_test, y_test, ili_max = data(window=window, 
                                            gamma=gamma, 
                                            batch_size=batch_size, 
                                            year=year, 
                                            rescale = True, 
                                            country = country, 
                                            dtype=dtype)

    x_train = [x_tr.to(device) for x_tr in x_train]
    y_train = [y_tr.to(device) for y_tr in y_train]

    enc = models.Encoder_z0_RNN(latent_dim, input_dim, rnn_hidden_size=rnn_hidden_size, Sstddev = Sstddev,Istddev=Istddev, Fp = True)
    ode = models.Fp(latent_dim, nhidden=nhidden)
    dec = models.Decoder(latent_dim, input_dim, Fp = True)

    enc.to(device)
    ode.to(device)
    dec.to(device)

    t = torch.arange(gamma+window, dtype=dtype, device=device)

    _history = train_functions.history()
    optimizer = torch.optim.Adam(chain(enc.parameters(), ode.parameters(), dec.parameters()), lr=lr)

    # Pre training encoder - v handy
    pbar = tqdm.trange(150, disable = disable)
    for epoch in pbar:
        for x_tr, y_tr in zip(x_train, y_train):
            optimizer.zero_grad()

            mean, std = enc(x_tr)
            prior = train_functions.make_prior(mean, 'Fp', I_sig = Istddev, latent_dim = latent_dim)

            kl_z = train_functions.kl_divergence(prior, Normal(mean, std)).sum(-1).mean() / len(x_train)
            kl_z.backward()
            optimizer.step()

    pbar = tqdm.trange(epochs, disable=disable)
    for epoch in pbar:
        epc = len(_history.epoch_history)
        for x_tr, y_tr in zip(x_train, y_train):
            z = torch.tensor(np.random.normal(0, 1, (n_samples, input_dim, x_tr.shape[0], latent_dim)), dtype=dtype, device=device)
            ode.params = []
            optimizer.zero_grad()

            mean, std = enc(x_tr)
            samples = enc.reparameterise(mean, std, z)
            latent = odeint(ode, samples, t, method='rk4', options=dict(step_size = 1.0))
            y_pred = dec(latent)

            nll = train_functions.nll_loss(y_pred, y_tr)
            kl_p = klp_w * train_functions.get_kl_params(epc, ode.posterior(), limit = limit)
            kl_z = klz_w * train_functions.kl_divergence(train_functions.make_prior(mean, 'Fp', I_sig = Istddev), 
                                                 Normal(mean, std)).sum(-1).mean() / len(x_train)
            reg_loss = train_functions.latent_init_loss(latent[..., :3])

            loss = nll+kl_p+kl_z+reg_loss
            loss.backward()
            optimizer.step()
            _history.batch([loss.cpu(), nll.cpu(), kl_z.cpu(),kl_p.cpu(),reg_loss.cpu(), optimizer.param_groups[-1]['lr']], ['loss', 'nll', 'kl_latent', 'kl_params', 'reg_loss', 'lr'])
        _history.reset()
        utils.update_learning_rate(optimizer, lr_scale, lr/10)
        pbar.set_postfix(_history.epoch_history[-1])

    x_test = x_test.to(device)
    z = torch.tensor(np.random.normal(0, 1, (n_samples, input_dim, x_test.shape[0], latent_dim)), dtype=dtype, device=device)
    mean, std = enc(x_test)
    samples = enc.reparameterise(mean, std, z)
    latent = odeint(ode, samples, t, method='rk4', options=dict(step_size = 1.0))
    y_pred = dec(latent[..., :3])

    validation_loss = train_functions.nll_loss(y_pred, y_test)
    return float(validation_loss.detach().numpy())
                              
if __name__ == "__main__":
    import time
    import random
    from filelock import FileLock
    import sys
    import time
    import argparse

    position = int(sys.argv[1])
    lock = FileLock("Fp_validation_scores.csv.lock")
    for _ in range(256):
        with lock:
            df = pd.read_csv('Fp_validation_scores.csv', index_col=0)
            try:
                param_num = np.min(np.where(df['started'] == 0))
            except:
                print('oh no')
            df.loc[param_num, 'started'] = 1
            df.to_csv('Fp_validation_scores.csv')

        score = 10
        try:
            print('starting:', df.loc[param_num])
            score = evaluate(dict(df.loc[param_num]), disable=True)
        except Exception as e:
            print("An error occurred:", e)


        with lock:
            df = pd.read_csv('Fp_validation_scores.csv', index_col=0)
            df.loc[param_num, 'score'] = score
            df.to_csv('Fp_validation_scores.csv')