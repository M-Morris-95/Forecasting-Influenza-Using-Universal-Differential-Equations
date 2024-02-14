import pandas as pd
import os
import torch
from torch import nn
from itertools import chain
from torch.distributions import Normal
import numpy as np
import lib.utils as utils

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
        