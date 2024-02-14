import pandas as pd
import os
import torch
from torch import nn
from itertools import chain
from torch.distributions import Normal
import lib.utils as utils

def make_prior(mean, z_prior=torch.tensor([0.1, 0.01]), device='cpu', latent_dim=8):
    z_prior=z_prior.to(device)
    mean_concat = torch.cat((mean[..., :2]  , torch.zeros_like(mean[..., 2:], device=device)), dim=-1)
    std = torch.cat([z_prior[0].unsqueeze(0), z_prior[1].unsqueeze(0), torch.ones(latent_dim - len(z_prior) - 1, device=device)], 0).expand_as(mean_concat)

    return Normal(mean_concat, torch.abs(std))

def reparam(eps, std, mean, n_samples, batch_size):
    z = eps * std + mean
    z = torch.concat([torch.abs(z[..., :2]), (1 - torch.abs(z[..., :2]).sum(-1)).unsqueeze(-1), z[..., 2:]], -1)
    z = z.reshape((n_samples * batch_size, ) + z.shape[2:])
    return z

class Decoder(nn.Module):
    def __init__(self, n_regions, latent_dim, input_dim, Fp=True, device=torch.device("cpu"), dtype=torch.float32):
        super(Decoder, self).__init__()
        self.Fp = Fp
        self.n_regions = n_regions
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if self.Fp:
            self.latent_dim = 3
            
        decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_regions*self.latent_dim, n_regions*input_dim)
        ).to(device=device, dtype=dtype)
        
        utils.init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        
        data = data[..., :self.latent_dim]
        shape = data.shape
        data = data.reshape((-1, )+ tuple(shape[2:]))
        out = self.decoder(data)
        return out.reshape(tuple(shape[:2]) + (-1,))
    

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

class Fp(nn.Module):
    def __init__(self, n_region=1, latent_dim=8, nhidden=20):
        super(Fp, self).__init__()

        self.n_region = n_region
        self.latent_dim = latent_dim
        self.ode_type = 'Fp'
        self.uncertainty = 'none'

        self.Fp_net = nn.Sequential(
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
        out = torch.abs(self.Fp_net(x)).reshape(-1, self.n_region, 2)
        
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
 
class Fa(nn.Module):
    def __init__(self, n_regions=1, latent_dim=8, net_sizes=[32, 32], aug_net_sizes=[32, 32], nhidden_fa=32):
        super(Fa, self).__init__()
        self.ode_type = 'Fa'
        self.uncertainty = 'none'
        self.n_regions = n_regions
        self.latent_dim = latent_dim
        self.flatten = nn.Flatten()

        self.aug_net = nn.ModuleList()
        self.aug_net.append(nn.Linear(n_regions * latent_dim, aug_net_sizes[0]))
        for l in range(1, len(aug_net_sizes)):
            self.aug_net.append(nn.ELU(inplace=True))
            self.aug_net.append(nn.Linear(aug_net_sizes[l - 1], aug_net_sizes[l]))
        self.aug_net.append(nn.Linear(aug_net_sizes[-1], 3 * n_regions))

        self.params = []
        self.tracker = []

    def forward(self, t, x):
        out_of_range_mask = (x > 2) | (x < -1)

        out_aug = self.flatten(x)
        for layer in self.aug_net:
            out_aug = layer(out_aug)
        Fa = out_aug.reshape(-1, self.n_regions, 3)
        res = torch.cat([Fa, torch.zeros_like(x[..., 3:])], -1)

        res[out_of_range_mask] = 0.0
        self.tracker.append(Fa)
        return res

    def clear_tracking(self):
        self.params = []
        self.tracker = []

    def posterior(self):
        params = torch.stack(self.params).reshape(-1, 2)
        self.params = []
        return Normal(params.mean(0), params.std(0))

class FaFp(nn.Module):
    def __init__(self, n_regions=1, latent_dim=8, nhidden=20, aug_net_sizes=[32, 32]):
        super(FaFp, self).__init__()

        self.n_regions = n_regions
        self.latent_dim = latent_dim
        self.ode_type = 'FaFp'
        self.uncertainty = 'none'

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_regions*latent_dim, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, 2*n_regions),
        )

        self.aug_net = nn.ModuleList()
        self.aug_net.append(nn.Flatten())
        self.aug_net.append(nn.Linear(n_regions * latent_dim, aug_net_sizes[0]))
        for l in range(1, len(aug_net_sizes)):
            self.aug_net.append(nn.ELU(inplace=True))
            self.aug_net.append(nn.Linear(aug_net_sizes[l - 1], aug_net_sizes[l]))
        self.aug_net.append(nn.Linear(aug_net_sizes[-1], 3 * n_regions))
        
        self.Fa_w = 1.0

        self.params = []
        self.tracker = []

    def forward(self, t, x):
        out_of_range_mask = (x > 2) | (x < -1)
        out = torch.abs(self.net(x)).reshape(-1, self.n_regions, 2)
        
        self.params.append(out)

        plusI = out[..., 0] * x[..., 0] * x[..., 1]
        minusI = out[..., 1] * x[..., 1]

        Fp = torch.stack([-plusI, plusI - minusI, minusI], dim=-1)
        
        out_aug = x
        for layer in self.aug_net:
            out_aug = layer(out_aug)
        Fa = out_aug.reshape(-1, self.n_regions, 3)
        res = torch.cat([Fp + self.Fa_w * Fa, torch.zeros_like(x[..., 3:])], -1)

        res[out_of_range_mask] = 0.0
        self.tracker.append(Fa)

        return res
    
    def clear_tracking(self):
        """Resets the trackers."""
        self.params = []
        self.tracker = []
    
    def posterior(self):
        params = torch.stack(self.params).reshape(-1, 2)
        
        self.params = []
        return Normal(params.mean(0), params.std(0))