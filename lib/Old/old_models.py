import pandas as pd
import os
import torch
from torch import nn
from itertools import chain
from torch.distributions import Normal
import lib.utils as utils

class Encoder(nn.Module):
    def __init__(self, rnn_input_size, rnn_hidden_sizes, ff_hidden_sizes, n_regions, latent_dim, SIR_scaler = [0.1, 0.05, 1.0], device='cpu', dtype=torch.float32):
        super(Encoder, self).__init__()

        ff_hidden_sizes.append(2*n_regions*latent_dim)
        self.device=device
        self.dtype = dtype
        self.n_regions = n_regions
        self.latent_dim = latent_dim
        self.scaler = torch.tensor(SIR_scaler, dtype=dtype, device=device)
        if latent_dim > len(self.scaler):
            extension = self.scaler[-1].repeat(latent_dim - len(self.scaler))
            self.scaler = torch.cat([self.scaler, extension])
        self.scaler = self.scaler.view(1, 1, -1)
            
        self.rnn = CustomRNN(rnn_input_size, rnn_hidden_sizes, device=device, dtype=dtype)
        self.ff = FFNetwork(rnn_hidden_sizes[-1], ff_hidden_sizes, device=device, dtype=dtype)

    def forward(self, x, reverse=True):
        if reverse:
            x = torch.flip(x, [1])
            
        rnn_output = self.rnn(x)
        ff_output = self.ff(rnn_output)
        mean, std = utils.split_last_dim(ff_output)

        mean = mean.reshape(-1, self.n_regions, self.latent_dim)
        std = std.reshape(-1, self.n_regions, self.latent_dim)
        
        std = torch.abs(std) * self.scaler
        return mean, std


class Encoder_MISO_GRU(nn.Module):
    def __init__(self, n_regions, input_size=10, latent_dim = 6, q_sizes=[128, 64], ili_sizes=[32, 16], ff_sizes = [64,32], SIR_scaler=[0.1, 0.05, 1.0], device='cpu', dtype=torch.float32):
        super(Encoder_MISO_GRU, self).__init__()

        n_qs = input_size-1
        self.scaler = torch.tensor(SIR_scaler, dtype=dtype, device=device)
        self.latent_dim = latent_dim
        if latent_dim > len(self.scaler):
            extension = self.scaler[-1].repeat(latent_dim - len(self.scaler))
            self.scaler = torch.cat([self.scaler, extension])
        self.scaler = self.scaler.view(1, -1)
        
        self.n_regions = n_regions

        self.i_layers = nn.ModuleList()
        self.i_layers.append(nn.GRU(n_regions, ili_sizes[0], batch_first=True))
        for l in range(1, len(ili_sizes)):
            self.i_layers.append(nn.GRU(ili_sizes[l-1], ili_sizes[l], batch_first=True))

        self.q_layers = nn.ModuleList()
        self.q_layers.append(nn.GRU(n_regions * n_qs, q_sizes[0], bidirectional=True, batch_first=True))
        for l in range(1, len(q_sizes)):
            self.q_layers.append(nn.GRU(2*q_sizes[l-1], q_sizes[l], bidirectional=True, batch_first=True))

        self.ff_layers = nn.ModuleList()
        self.ff_layers.append(nn.Linear(2*q_sizes[-1] + ili_sizes[-1], ff_sizes[0]))
        for l in range(1, len(ff_sizes)):
            self.ff_layers.append(nn.ReLU())
            self.ff_layers.append(nn.Linear(ff_sizes[l-1], ff_sizes[l]))
        self.ff_layers.append(nn.Linear(ff_sizes[l], 2 * n_regions * latent_dim))

    def forward(self, x):
        x_qs = x[:, :, :-self.n_regions]
        x_ili = x[:, :-14, -self.n_regions:]

        for GRU_layer in self.i_layers:
            x_ili, _ = GRU_layer(x_ili)

        for GRU_layer in self.q_layers:
            x_qs, _ = GRU_layer(x_qs)
        x_concat = torch.cat([x_ili[:, -1, :], x_qs[:, -1, :]], -1)

        for ff_layer in self.ff_layers:
            x_concat = ff_layer(x_concat)

        mean, std = utils.split_last_dim(x_concat)
        mean = mean.reshape(-1, self.n_regions, self.latent_dim)
        std = std.reshape(-1, self.n_regions, self.latent_dim)
        std = torch.abs(std) * self.scaler
        return mean, std

class Encoder_BiDirectionalLSTM(nn.Module):
    def __init__(self, n_regions, n_qs=10, latent_dim = 6, q_sizes=[128, 64], ili_sizes=[32, 16], ff_sizes = [64,32], SIR_scaler=[0.1, 0.05, 1.0], device='cpu', dtype=torch.float32):
        super(Encoder_BiDirectionalLSTM, self).__init__()

        self.scaler = torch.tensor(SIR_scaler, dtype=dtype, device=device)
        self.latent_dim = latent_dim
        if latent_dim > len(self.scaler):
            extension = self.scaler[-1].repeat(latent_dim - len(self.scaler))
            self.scaler = torch.cat([self.scaler, extension])
        self.scaler = self.scaler.view(1, -1)
        
        self.n_regions = n_regions

        self.i_layers = nn.ModuleList()
        self.i_layers.append(nn.GRU(n_regions, ili_sizes[0], batch_first=True))
        for l in range(1, len(ili_sizes)):
            self.i_layers.append(nn.GRU(ili_sizes[l-1], ili_sizes[l], batch_first=True))

        self.q_layers = nn.ModuleList()
        self.q_layers.append(nn.GRU(n_regions * n_qs, q_sizes[0], bidirectional=True, batch_first=True))
        for l in range(1, len(q_sizes)):
            self.q_layers.append(nn.GRU(q_sizes[l-1], q_sizes[l], bidirectional=True, batch_first=True))

        self.ff_layers = nn.ModuleList()
        self.ff_layers.append(nn.Linear(q_sizes[-1]*2 + ili_sizes[-1], ff_sizes[0]))
        for l in range(1, len(ff_sizes)):
            self.ff_layers.append(nn.ReLU())
            self.ff_layers.append(nn.Linear(ff_sizes[l-1], ff_sizes[l]))
        self.ff_layers.append(nn.Linear(ff_sizes[l], 2 * n_regions * latent_dim))

    def forward(self, x):
        x_qs = x[:, :, :-self.n_regions]
        x_ili = x[:, :-14, -self.n_regions:]

        for lstm_layer in self.i_layers:
            x_ili, _ = lstm_layer(x_ili)

        for lstm_layer in self.q_layers:
            x_qs, _ = lstm_layer(x_qs)

        x_concat = torch.cat([x_ili[:, -1, :], x_qs[:, -1, :]], -1)

        for ff_layer in self.ff_layers:
            x_concat = ff_layer(x_concat)

        mean, std = utils.split_last_dim(x_concat)
        mean = mean.reshape(-1, self.n_regions, self.latent_dim)
        std = std.reshape(-1, self.n_regions, self.latent_dim)
        std = torch.abs(std) * self.scaler
        return mean, std

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, device='cpu', dtype=torch.float32):
        super(CustomRNN, self).__init__()
        self.layers = nn.ModuleList()
        self.device = device
        self.dtype = dtype

        # Creating the LSTM layers
        for i, hidden_size in enumerate(hidden_sizes):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            lstm_layer = nn.GRU(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
            self.layers.append(lstm_layer.to(device=device, dtype=dtype))

    def forward(self, x):
        x = x.to(device=self.device, dtype=self.dtype)
        # Forward pass through LSTM layers
        for layer in self.layers:
            x, _ = layer(x)
        
        return x[:, -1, :]

class FFNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, device='cpu', dtype=torch.float32):
        super(FFNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.device = device
        self.dtype = dtype

        # Creating the dense layers
        for layer_num, hidden_size in enumerate(hidden_sizes):
            dense_layer = nn.Linear(input_size, hidden_size).to(device=device, dtype=dtype)
            relu_layer = nn.ReLU().to(device=device, dtype=dtype)
            self.layers.append(dense_layer)

            if layer_num != len(hidden_sizes)-1:
                self.layers.append(relu_layer)
            input_size = hidden_size

    def forward(self, x):
        x = x.to(device=self.device, dtype=self.dtype)
        for layer in self.layers:
            x = layer(x)
        return x


# import torch
# import torch.nn as nn
# from torch.distributions import Normal

# class FaFp(nn.Module):
#     def __init__(self, n_regions=1, latent_dim=8, net_sizes=[32, 32], aug_net_sizes=[32, 32], nhidden_fa=32):
#         super(FaFp, self).__init__()
#         self.n_regions = n_regions
#         self.latent_dim = latent_dim
#         self.flatten = nn.Flatten()
            
#         self.net = nn.ModuleList()
#         self.net.append(nn.Linear(n_regions * latent_dim, net_sizes[0]))
#         for l in range(1, len(net_sizes)):
#             self.net.append(nn.ELU(inplace=True))
#             self.net.append(nn.Linear(net_sizes[l - 1], net_sizes[l]))
#         self.net.append(nn.Linear(net_sizes[-1], 2 * n_regions))

#         self.aug_net = nn.ModuleList()
#         self.aug_net.append(nn.Linear(n_regions * latent_dim, aug_net_sizes[0]))
#         for l in range(1, len(aug_net_sizes)):
#             self.aug_net.append(nn.ELU(inplace=True))
#             self.aug_net.append(nn.Linear(aug_net_sizes[l - 1], aug_net_sizes[l]))
#         self.aug_net.append(nn.Linear(aug_net_sizes[-1], 3 * n_regions))

#         self.params = []
#         self.tracker = []

#         self.ode_type = 'FaFp'

#     def forward(self, t, x):
#         out_of_range_mask = (x > 2) | (x < -1)

#         # Applying layers in self.net sequentially
#         out = self.flatten(x)
        
#         for layer in self.net:
#             out = layer(out)
#         out = torch.abs(out).reshape(-1, self.n_regions, 2)
        
#         plusI = out[..., 0] * x[..., 0] * x[..., 1]
#         minusI = out[..., 1] * x[..., 1]

#         # Applying layers in self.aug_net sequentially
#         Fp = torch.stack([-plusI, plusI - minusI, minusI], dim=-1)
#         out_aug = self.flatten(x)
#         for layer in self.aug_net:
#             out_aug = layer(out_aug)
#         Fa = out_aug.reshape(-1, self.n_regions, 3)
        
#         res = torch.cat([Fp + Fa, torch.zeros_like(x[..., 3:])], -1)

#         res[out_of_range_mask] = 0.0
#         self.tracker.append(Fa)
#         self.params.append(out)

#         return res

#     def clear_tracking(self):
#         self.params = []
#         self.tracker = []

#     def posterior(self):
#         params = torch.stack(self.params).reshape(-1, 2)
#         self.params = []
#         return Normal(params.mean(0), params.std(0))


import torch
import torch.nn as nn
from torch.distributions import Normal

import torch
import torch.nn as nn
from torch.distributions import Normal

import lib.utils as utils

class Encoder_Back_GRU(nn.Module):
    def __init__(self, n_regions, n_qs=9, latent_dim = 6, q_sizes=[128, 64], ff_sizes = [64,32], ili_sizes=None, SIR_scaler=[0.1, 0.05, 1.0], device='cpu', dtype=torch.float32):
        super(Encoder_Back_GRU, self).__init__()
        self.latent_dim = latent_dim
        input_size = n_qs+1
        self.n_regions = n_regions
        
        self.scaler = torch.tensor(SIR_scaler, dtype=dtype, device=device)
        if latent_dim > len(self.scaler):
            extension = self.scaler[-1].repeat(latent_dim - len(self.scaler))
            self.scaler = torch.cat([self.scaler, extension])
        self.scaler = self.scaler.view(1, -1)
        
        
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(nn.GRU(n_regions * input_size, q_sizes[0], batch_first=True))
        for l in range(1, len(q_sizes)):
            self.rnn_layers.append(nn.GRU(q_sizes[l-1], q_sizes[l], batch_first=True))

        # self.ff_layers = nn.ModuleList()
        # self.ff_layers.append(nn.Linear(q_sizes[-1], ff_sizes[0]))
        # for l in range(1, len(ff_sizes)):
        #     self.ff_layers.append(nn.ReLU())
        #     self.ff_layers.append(nn.Linear(ff_sizes[l-1], ff_sizes[l]))
        # self.ff_layers.append(nn.Linear(ff_sizes[l], 2 * n_regions * latent_dim))

        self.ff_layers = nn.ModuleList()
        self.ff_layers.append(nn.Linear(q_sizes[-1], ff_sizes[0]))
        if len(ff_sizes) > 1:
            for l in range(1, len(ff_sizes)):
                self.ff_layers.append(nn.ReLU())
                self.ff_layers.append(nn.Linear(ff_sizes[l-1], ff_sizes[l]))
        else:
            l = 0
        self.ff_layers.append(nn.Linear(ff_sizes[l], 2 * n_regions * latent_dim))

    def forward(self, x):
        x = x.flip(1)

        for GRU_layer in self.rnn_layers:
            x, _ = GRU_layer(x)
        
        x = x[:, -1, :]
        for ff_layer in self.ff_layers:
            x = ff_layer(x)

        mean, std = utils.split_last_dim(x)
        mean = mean.reshape(-1, self.n_regions, self.latent_dim)
        std = std.reshape(-1, self.n_regions, self.latent_dim)
        std = torch.abs(std) * self.scaler
        return mean, std

class Encoder_MISO_GRU(nn.Module):
    def __init__(self, n_regions, n_qs=9, latent_dim = 6, q_sizes=[128, 64], ili_sizes=[32, 16], ff_sizes = [64,32], SIR_scaler=[0.1, 0.05, 1.0], device='cpu', dtype=torch.float32):
        super(Encoder_MISO_GRU, self).__init__()
        input_size = n_qs+1
        self.scaler = torch.tensor(SIR_scaler, dtype=dtype, device=device)
        self.latent_dim = latent_dim
        if latent_dim > len(self.scaler):
            extension = self.scaler[-1].repeat(latent_dim - len(self.scaler))
            self.scaler = torch.cat([self.scaler, extension])
        self.scaler = self.scaler.view(1, -1)
        
        self.n_regions = n_regions

        self.i_layers = nn.ModuleList()
        self.i_layers.append(nn.GRU(n_regions, ili_sizes[0], batch_first=True))
        for l in range(1, len(ili_sizes)):
            self.i_layers.append(nn.GRU(ili_sizes[l-1], ili_sizes[l], batch_first=True))

        self.q_layers = nn.ModuleList()
        self.q_layers.append(nn.GRU(n_regions * n_qs, q_sizes[0], bidirectional=True, batch_first=True))
        for l in range(1, len(q_sizes)):
            self.q_layers.append(nn.GRU(2*q_sizes[l-1], q_sizes[l], bidirectional=True, batch_first=True))

        self.ff_layers = nn.ModuleList()
        self.ff_layers.append(nn.Linear(2*q_sizes[-1] + ili_sizes[-1], ff_sizes[0]))
        if len(ff_sizes) > 1:
            for l in range(1, len(ff_sizes)):
                self.ff_layers.append(nn.ReLU())
                self.ff_layers.append(nn.Linear(ff_sizes[l-1], ff_sizes[l]))
        else:
            l = 0
        self.ff_layers.append(nn.Linear(ff_sizes[l], 2 * n_regions * latent_dim))

    def forward(self, x):
        x_qs = x[:, :, :-self.n_regions]
        x_ili = x[:, :-14, -self.n_regions:]

        for GRU_layer in self.i_layers:
            x_ili, _ = GRU_layer(x_ili)

        for GRU_layer in self.q_layers:
            x_qs, _ = GRU_layer(x_qs)
        x_concat = torch.cat([x_ili[:, -1, :], x_qs[:, -1, :]], -1)

        for ff_layer in self.ff_layers:
            x_concat = ff_layer(x_concat)

        mean, std = utils.split_last_dim(x_concat)
        mean = mean.reshape(-1, self.n_regions, self.latent_dim)
        std = std.reshape(-1, self.n_regions, self.latent_dim)
        std = torch.abs(std) * self.scaler
        return mean, std

class Encoder_BiDirectionalGRU(nn.Module):
    def __init__(self, n_regions, n_qs=10, latent_dim = 6, q_sizes=[128, 64], ili_sizes=[32, 16], ff_sizes = [64,32], SIR_scaler=[0.1, 0.05, 1.0], device='cpu', dtype=torch.float32):
        super(Encoder_BiDirectionalGRU, self).__init__()

        self.scaler = torch.tensor(SIR_scaler, dtype=dtype, device=device)
        self.latent_dim = latent_dim
        if latent_dim > len(self.scaler):
            extension = self.scaler[-1].repeat(latent_dim - len(self.scaler))
            self.scaler = torch.cat([self.scaler, extension])
        self.scaler = self.scaler.view(1, -1)
        
        self.n_regions = n_regions

        self.i_layers = nn.ModuleList()
        self.i_layers.append(nn.GRU(n_regions, ili_sizes[0], batch_first=True))
        for l in range(1, len(ili_sizes)):
            self.i_layers.append(nn.GRU(ili_sizes[l-1], ili_sizes[l], batch_first=True))

        self.q_layers = nn.ModuleList()
        self.q_layers.append(nn.GRU(n_regions * n_qs, q_sizes[0], bidirectional=True, batch_first=True))
        for l in range(1, len(q_sizes)):
            self.q_layers.append(nn.GRU(2*q_sizes[l-1], q_sizes[l], bidirectional=True, batch_first=True))



        self.ff_layers = nn.ModuleList()
        self.ff_layers.append(nn.Linear(q_sizes[-1]*2 + ili_sizes[-1], ff_sizes[0]))
        if len(ff_sizes) > 1:
            for l in range(1, len(ff_sizes)):
                self.ff_layers.append(nn.ReLU())
                self.ff_layers.append(nn.Linear(ff_sizes[l-1], ff_sizes[l]))
        else:
            l = 0
        self.ff_layers.append(nn.Linear(ff_sizes[l], 2 * n_regions * latent_dim))



        # self.ff_layers = nn.ModuleList()
        # self.ff_layers.append(nn.Linear(q_sizes[-1]*2 + ili_sizes[-1], ff_sizes[0]))
        # for l in range(1, len(ff_sizes)):
        #     self.ff_layers.append(nn.ReLU())
        #     self.ff_layers.append(nn.Linear(ff_sizes[l-1], ff_sizes[l]))
        # self.ff_layers.append(nn.Linear(ff_sizes[l], 2 * n_regions * latent_dim))

    def forward(self, x):
        x_qs = x[:, :, :-self.n_regions]
        x_ili = x[:, :-14, -self.n_regions:]

        for GRU_layer in self.i_layers:
            x_ili, _ = GRU_layer(x_ili)

        for GRU_layer in self.q_layers:
            x_qs, _ = GRU_layer(x_qs)

        x_concat = torch.cat([x_ili[:, -1, :], x_qs[:, -1, :]], -1)

        for ff_layer in self.ff_layers:
            x_concat = ff_layer(x_concat)

        mean, std = utils.split_last_dim(x_concat)
        mean = mean.reshape(-1, self.n_regions, self.latent_dim)
        std = std.reshape(-1, self.n_regions, self.latent_dim)
        std = torch.abs(std) * self.scaler
        return mean, std

def reparam(eps, std, mean, n_samples, batch_size):
    z = eps * std + mean
    z = torch.concat([torch.abs(z[..., :2]), (1 - torch.abs(z[..., :2]).sum(-1)).unsqueeze(-1), z[..., 2:]], -1)
    z = z.reshape((n_samples * batch_size, ) + z.shape[2:])
    return z

def make_prior(mean, z_prior=torch.tensor([0.1, 0.01]), device='cpu', latent_dim=8):
    z_prior=z_prior.to(device)
    mean_concat = torch.cat((mean[..., :2]  , torch.zeros_like(mean[..., 2:], device=device)), dim=-1)
    std = torch.cat([z_prior[0].unsqueeze(0), z_prior[1].unsqueeze(0), torch.ones(latent_dim - len(z_prior) - 1, device=device)], 0).expand_as(mean_concat)

    return Normal(mean_concat, torch.abs(std))



import torch
import torch.nn as nn
from torch.distributions import Normal

class Encoder_Back_GRU(nn.Module):
    def __init__(self, n_regions, n_qs=9, latent_dim = 6, q_sizes=[128, 64], ff_sizes = [32], ili_sizes=None, SIR_scaler=[0.1, 0.05, 1.0], device='cpu', dtype=torch.float32):
        super(Encoder_Back_GRU, self).__init__()
        self.latent_dim = latent_dim
        input_size = n_qs+1
        self.n_regions = n_regions
        self.device = device
        self.dtype = dtype
        
        self.scaler = torch.tensor(SIR_scaler, dtype=self.dtype, device=self.device)
        if latent_dim > len(self.scaler):
            extension = self.scaler[-1].repeat(latent_dim - len(self.scaler))
            self.scaler = torch.cat([self.scaler, extension])
        self.scaler = self.scaler.view(1, -1)
        
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(nn.GRU(n_regions * input_size, q_sizes[0], batch_first=True))
        for l in range(1, len(q_sizes)):
            self.rnn_layers.append(nn.GRU(q_sizes[l-1], q_sizes[l], batch_first=True))

        self.ff_layers = nn.ModuleList()
        self.ff_layers.append(nn.Linear(q_sizes[-1], ff_sizes[0]))
        if len(ff_sizes) > 1:
            for l in range(1, len(ff_sizes)):
                self.ff_layers.append(nn.ReLU())
                self.ff_layers.append(nn.Linear(ff_sizes[l-1], ff_sizes[l]))
        else:
            l = 0
        self.ff_layers.append(nn.Linear(ff_sizes[l], 2 * n_regions * latent_dim))

    def forward(self, x):
        x = x.flip(1)

        for GRU_layer in self.rnn_layers:
            x, _ = GRU_layer(x)
        
        x = x[:, -1, :]
        for ff_layer in self.ff_layers:
            x = ff_layer(x)

        mean, std = torch.split(x, split_size_or_sections=x.size(-1) // 2, dim=-1)

        mean = mean.reshape(-1, self.n_regions, self.latent_dim)
        std = std.reshape(-1, self.n_regions, self.latent_dim)

        std = torch.abs(std) * self.scaler
        return mean, std
    

    