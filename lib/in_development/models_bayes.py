import pandas as pd
import os
import torch
from torch import nn
from itertools import chain
from torch.distributions import Normal
import lib.utils as utils
import math
import torch.distributions as dist


class Dense_Variational(nn.Module):
    def __init__(self, in_features, out_features, bias=True, prior_std=1.0):
        super(Dense_Variational, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        self.w_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_std = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias=True
            self.b_mean = nn.Parameter(torch.Tensor(out_features))
            self.b_std = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def make_z(self, ):
        with torch.no_grad():
            self.z = [torch.randn_like(self.w_mean), torch.randn_like(self.b_mean)]

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_mean, a=math.sqrt(5))
        nn.init.constant_(self.w_std, 0.1)  # Initialize std deviation
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_mean)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_mean, -bound, bound)
            nn.init.constant_(self.b_std, 0.1)

    def forward(self, input):
        self.make_z()
        w = self.w_mean + self.z[0]*torch.abs(self.w_std)
        b = self.b_mean + self.z[1]*torch.abs(self.b_std)

        return nn.functional.linear(input, w, b)
    
    def make_prior(self):
        self.prior = [
            dist.Normal(torch.zeros_like(self.w_mean), self.prior_std * torch.ones_like(self.w_mean)),
            dist.Normal(torch.zeros_like(self.b_mean), self.prior_std * torch.ones_like(self.b_mean)),
        ]
        return self.prior

    def make_posterior(self):
        self.posterior = [
            dist.Normal(self.w_mean, torch.abs(self.w_std)),
            dist.Normal(self.b_mean, torch.abs(self.b_std)),
        ]
        return self.posterior

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Bayes_Fp(nn.Module):
    def __init__(self, n_regions=1, latent_dim=8, net_sizes=[20, 20], prior_std=0.1, **kwargs):
        super(Bayes_Fp, self).__init__()

        self.n_regions = n_regions
        self.latent_dim = latent_dim
        self.ode_type = 'Fp'
        self.uncertainty = 'bayes'

        self.Fp_net = nn.ModuleList()
        self.Fp_net.append(nn.Flatten())
        self.Fp_net.append(Dense_Variational(n_regions * latent_dim, net_sizes[0], prior_std=prior_std))
        for l in range(1, len(net_sizes)):
            self.Fp_net.append(nn.ELU(inplace=True))
            self.Fp_net.append(Dense_Variational(net_sizes[l - 1], net_sizes[l], prior_std=prior_std))
        self.Fp_net.append(Dense_Variational(net_sizes[-1], 2 * n_regions, prior_std=prior_std))
        
        self.params = []

    def forward(self, t, x):
        out_of_range_mask = (x > 2) | (x < -1)

        out = x
        for layer in self.Fp_net:
            out = layer(out)
        out = torch.abs(out).reshape(-1, self.n_regions, 2)
        
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
 
 
    def get_kl(self):
        kl = 0
        count = 0
        for layer in list(self.children())[0]:
            try:
                kl = kl + sum([dist.kl_divergence(q, p).mean() for q, p in zip(layer.make_posterior(), layer.make_prior())])/2
                count = count + 1
            except:
                pass
        kl = kl/count
        return kl


class Bayes_Fa(nn.Module):
    def __init__(self, n_regions=1, latent_dim=8, aug_net_sizes=[32, 32], nhidden_fa=32, prior_std=0.1, **kwargs):
        super(Bayes_Fa, self).__init__()
        self.ode_type = 'Fa'
        self.uncertainty = 'bayes'

        self.n_regions = n_regions
        self.latent_dim = latent_dim
        
        self.aug_net = nn.ModuleList()
        self.aug_net.append(nn.Flatten())
        self.aug_net.append(Dense_Variational(n_regions * latent_dim, aug_net_sizes[0], prior_std=prior_std))
        for l in range(1, len(aug_net_sizes)):
            self.aug_net.append(nn.ELU(inplace=True))
            self.aug_net.append(Dense_Variational(aug_net_sizes[l - 1], aug_net_sizes[l], prior_std=prior_std))
        self.aug_net.append(Dense_Variational(aug_net_sizes[-1], 3 * n_regions, prior_std=prior_std))

        self.params = []
        self.tracker = []

    def forward(self, t, x):
        out_of_range_mask = (x > 2) | (x < -1)

        out = x
        for layer in self.aug_net:
            out = layer(out)
        Fa = out.reshape(-1, self.n_regions, 3)
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

    def get_kl(self):
        kl = 0
        count = 0
        for layer in list(self.children())[0]:
            try:
                kl = kl + sum([dist.kl_divergence(q, p).mean() for q, p in zip(layer.make_posterior(), layer.make_prior())])/2
                count = count + 1
            except:
                pass
        kl = kl/count
        return kl

class Bayes_FaFp(nn.Module):
    def __init__(self, n_regions=1, latent_dim=8, net_sizes=[20, 20], aug_net_sizes=[32, 32], prior_std=0.1, **kwargs):
        super(Bayes_FaFp, self).__init__()

        self.n_regions = n_regions
        self.latent_dim = latent_dim
        self.ode_type = 'FaFp'
        self.uncertainty = 'bayes'

        self.Fp_net = nn.ModuleList()
        self.Fp_net.append(nn.Flatten())
        self.Fp_net.append(Dense_Variational(n_regions * latent_dim, net_sizes[0], prior_std=prior_std))
        for l in range(1, len(net_sizes)):
            self.Fp_net.append(nn.ELU(inplace=True))
            self.Fp_net.append(Dense_Variational(net_sizes[l - 1], net_sizes[l], prior_std=prior_std))
        self.Fp_net.append(Dense_Variational(net_sizes[-1], 2 * n_regions, prior_std=prior_std))


        self.aug_net = nn.ModuleList()
        self.aug_net.append(nn.Flatten())
        self.aug_net.append(Dense_Variational(n_regions * latent_dim, aug_net_sizes[0], prior_std=prior_std))
        for l in range(1, len(aug_net_sizes)):
            self.aug_net.append(nn.ELU(inplace=True))
            self.aug_net.append(Dense_Variational(aug_net_sizes[l - 1], aug_net_sizes[l], prior_std=prior_std))
        self.aug_net.append(Dense_Variational(aug_net_sizes[-1], 3 * n_regions, prior_std=prior_std))
        
        self.Fa_w = 1.0

        self.params = []
        self.tracker = []

    def forward(self, t, x):
        out_of_range_mask = (x > 2) | (x < -1)

        out = x
        for layer in self.Fp_net:
            out = layer(out)
        out = torch.abs(out).reshape(-1, self.n_regions, 2)
        
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
    
    def get_kl(self):
        kl = 0
        count = 0
        
        for model in list(self.children()):
            for layer in list(model):
                try:
                    kl = kl + sum([dist.kl_divergence(q, p).mean() for q, p in zip(layer.make_posterior(), layer.make_prior())])/2
                    count = count + 1
                except:
                    pass
        kl = kl/count
        return kl