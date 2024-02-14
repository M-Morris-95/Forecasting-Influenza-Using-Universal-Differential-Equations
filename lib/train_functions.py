import numpy as np
import torch
import torch.nn as nn
import os
from torch.nn.functional import relu
from torch.distributions import Categorical, Normal
from torch.nn.modules.rnn import LSTM, GRU
from torch.distributions import MultivariateNormal, Normal, kl_divergence
import subprocess

# import lib.Data_Constructor as Data_Constructor
def make_file(prefix):
    root = '/'.join(prefix.split('/')[:-1])
    if not os.path.exists(root):
        os.makedirs(root)
        
def KL_annealing(step, anneal_params):
    anneal = anneal_params.get('anneal', True)
    reset_pos = anneal_params.get('reset_pos', 10000)
    split = anneal_params.get('split', 0.5)
    lower = anneal_params.get('lower', 0.0)
    upper = anneal_params.get('upper', 1.0)
    anneal_type = anneal_params.get('type', 'linear')

    if not anneal:
        return 1

    while step > reset_pos:
        step -= reset_pos

    if step >= int(reset_pos * split):
        return upper

    frac = step / (int(reset_pos * split))
    if anneal_type == 'linear':
        return frac * (upper - lower) + lower

    if anneal_type == 'sigmoid':
        # Sigmoid annealing
        return lower + (upper - lower) / (1 + np.exp(-10 * (frac - 0.5)))

    if anneal_type == 'cosine':
        # Cosine annealing
        return lower + 0.5 * (1 - np.cos(np.pi * frac)) * (upper - lower)



    

# def prepare_data(test_season, data_season, gamma, window_size, n_queries, batch_size, dtype=torch.float32, device='cpu', selection_method='distance'):
#     # Initialize and call DataConstructor
#     _data = Data_Constructor.DataConstructor(test_season, data_season, gamma, window_size, n_queries=n_queries, selection_method=selection_method)
#     _data()
#     inputs, outputs, dates, test_dates = _data.build()

#     # Find the start index for the test data
#     test_start = np.where(dates == test_dates[0])[0][0]

#     # Convert data into tensors and move them to the specified device
#     x_tr = torch.tensor(inputs[:test_start], dtype=dtype).to(device)
#     y_tr = torch.tensor(outputs[:test_start], dtype=dtype).to(device)
#     x_test = torch.tensor(inputs[test_start:], dtype=dtype).to(device)
#     y_test = torch.tensor(outputs[test_start:], dtype=dtype).to(device)

#     # Batch the training data
#     n_batches = int(np.ceil(x_tr.shape[0] / batch_size))
#     x_train = []
#     y_train = []
#     for b in range(n_batches):
#         x_batch = x_tr[b*batch_size:(b+1)*batch_size].clone().detach()
#         y_batch = y_tr[b*batch_size:(b+1)*batch_size].clone().detach().unsqueeze(-2)
#         x_train.append(x_batch)
#         y_train.append(y_batch)

#     return x_train, y_train, x_test, y_test

def get_kl_params(epoch, Q, means=[0.8, 0.55], stds = [0.2, 0.2], device='cpu', limit = 1e10):
    if epoch < limit:
        return kl_divergence(Normal(torch.tensor(means, device=device), torch.tensor(stds, device=device)), Q).mean()
    return torch.tensor(0)
def nll_loss(y_pred, y, mean=True):
    y_std = torch.std(y_pred, 1)
    y_mean = torch.mean(y_pred, 1)
    nll = -Normal(y_mean, y_std).log_prob(y)
    mask = (y != -1).float()
    nll = nll * mask
    
    if mean:
        return nll.mean()
    return nll

def make_prior(mean, z_prior=torch.tensor([0.1, 0.01]), device='cpu', latent_dim=8):
    mean_concat = torch.cat((mean[..., :2]  , torch.zeros_like(mean[..., 2:], device=device)), dim=-1)
    std = torch.cat([z_prior[0].unsqueeze(0), z_prior[1].unsqueeze(0), torch.ones(latent_dim - len(z_prior) - 1, device=device)], 0).expand_as(mean_concat)

    return Normal(mean_concat, torch.abs(std))

def reparam(eps, std, mean, n_samples, batch_size):
    z = eps * std + mean
    z = torch.concat([torch.abs(z[..., :2]), (1 - torch.abs(z[..., :2]).sum(-1)).unsqueeze(-1), z[..., 2:]], -1)
    z = z.reshape((n_samples * batch_size, ) + z.shape[2:])
    return z
    
# def make_prior(mean, model_type, device='cpu', latent_dim = 8, stddev=0.1, S_sig=0.1, I_sig=0.1):
#     if model_type == 'Fa':
#         return Normal(torch.zeros_like(mean, device=device), stddev*torch.ones_like(mean, device=device))
#     elif model_type == 'SEIR':
#         mean = torch.concat([mean[..., :3], torch.zeros_like(mean[..., 3:], device=device)], -1)
#         std = torch.ones_like(mean, device=device) * torch.concat([torch.tensor([S_sig]), torch.tensor([I_sig]), torch.tensor([I_sig]), torch.ones(latent_dim-4)], 0)
#         return Normal(mean, std)
#     else:
#         mean = torch.concat([mean[..., :2], torch.zeros_like(mean[..., 2:], device=device)], -1)
#         std = torch.ones_like(mean, device=device) * torch.concat([torch.tensor([S_sig]), torch.tensor([I_sig]), torch.ones(latent_dim-3)], 0)
#         return Normal(mean, std)
    
def latent_init_loss(x):
    # Values where x < 0
    negative_values = torch.where(x < 0, abs(x), torch.zeros_like(x))
    
    # Values where x > 1
    greater_than_one_values = torch.where(x > 1, abs(1 - x), torch.zeros_like(x))
    
    # Combine the penalties
    total_loss = negative_values + greater_than_one_values
    
    return total_loss.sum()

def get_free_gpu():
    # Run the nvidia-smi command to get memory details
    sp = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()[0].decode('utf-8')

    # Extract memory details
    mem_details = [line.split(",") for line in out_str.split("\n") if line]
    total_memory = [int(detail[0].strip()) for detail in mem_details]
    used_memory = [int(detail[1].strip()) for detail in mem_details]
    free_memory = [total - used for total, used in zip(total_memory, used_memory)]

    # Return the GPU ID with the most free memory
    return free_memory.index(max(free_memory))

class history():
    def __init__(self):
        self.batches = []
        self.batch_history = []
        self.epoch_history = []
        return
    
    def batch(self,data=None, names=None, batch=None):
        if batch == None:
            batch = {}
            for d, name in zip(data, names):
                try:
                    batch[name] = d
                except:
                    batch[name] = d
                    
        for key in batch.keys():
            try:
                batch[key] = batch[key].detach().numpy()
            except:
                pass
        self.batches.append(batch)
        
        return
    
    def epoch(self):
        postfix = {}
        for key in self.batches[0].keys():
            postfix[key] = np.asarray([batch[key] for batch in self.batches]).mean()
        return postfix
    
    def reset(self):
        self.batch_history.append(self.batches)
        self.epoch_history.append(self.epoch())
        self.batches = []
        
def kl_div(Q_mu, Q_std, P_mu, P_std):
    return torch.sum(kl_divergence(Normal(Q_mu, Q_std), Normal(P_mu, P_std)), -1)

def test(x, y, input_dim=1, latent_dim=8, dtype=torch.float32, history = None, Fp=False):
    batch_size = x.shape[0]
    
    z = torch.tensor(np.random.normal(0, 1, (100, input_dim, batch_size, latent_dim)), dtype=dtype)

    mean, std = Enc(x.unsqueeze(-1), torch.arange(window))
    samples = mean + z*std
    if Fp:
        samples = torch.concat([torch.abs(samples), 1-torch.abs(samples).sum(-1).unsqueeze(-1)], -1)
    latent = odeint(Latent_func, samples, t)
    y_pred = Dec(latent).squeeze()

    nll = -torch.mean(torch.distributions.Normal(y_pred.mean(1), y_pred.std(1)).log_prob(y_tr.T))
    kl = kl_div(mean, std, torch.zeros(mean.shape, dtype=dtype), torch.ones(mean.shape, dtype=dtype))      


    mu = y_pred.mean(1).detach()
    std = y_pred.std(1).detach()
    for i in range(16):
        plt.subplot(4,4, i+1)
        plt.plot(mu[:, i], color='green', linewidth = 0.5)
        plt.fill_between(np.arange(gamma+window), (mu+std)[:, i], (mu-std)[:, i], color='green', alpha=0.3, linewidth = 0)
        plt.plot(y.T[:, i], color='black', linewidth = 0.5)
    plt.show()
    
    if history != None:
        for key in history.epoch_history[0].keys():
            plt.plot([h[key] for h in history.epoch_history], label = key)
        plt.legend()
        plt.show()
        
def validate(x, y, enc, dec, ode, t, samples=256, plot=False, dtype=torch.float32, device="cpu"):
    x = x.to(device)

    mean, std = enc(x)
    z = torch.tensor(np.random.normal(0, 1, (samples, x.shape[-1], x.shape[0], mean.shape[-1])), dtype=dtype, device=device)


    samples = enc.reparameterise(mean, std, z)
    latent = odeint(ode, samples, t, method='rk4', options=dict(step_size = 1.0))
    y_pred = dec(latent[..., :3])

    y_pred = y_pred.cpu()
    latent = latent.cpu()
    mean = mean.cpu()
    std = std.cpu()

    y_mean = y_pred.mean(1).squeeze().T.detach()
    y_std = y_pred.std(1).squeeze().T.detach()

    if plot:
    
        for g in range(5):
            plt.subplot(2,3,g+1)
            plt.plot(y_mean[:, g], color='red') 
            plt.fill_between(np.linspace(0, y_mean.shape[0]-1, y_mean.shape[0]), 
                             (y_mean + y_std)[:, g],
                             (y_mean - y_std)[:, g], color='red', linewidth = 0, alpha=0.3)
            plt.plot(y[:, g, 0], color='black')
            plt.title('$\gamma$ = ' + str(g+1-window))
        plt.show()
        for g in range(5, 9):
            plt.subplot(2,2,g-4)
            plt.plot(y_mean[:, g], color='red') 
            plt.fill_between(np.linspace(0, y_mean.shape[0]-1, y_mean.shape[0]), 
                             (y_mean + y_std)[:, g],
                             (y_mean - y_std)[:, g], color='red', linewidth = 0, alpha=0.3)
            plt.plot(y[:, g, 0], color='black')
            plt.title('$\gamma$ = '+ str(g+1-window))
        
    return float(torch.mean(-Normal(y_mean, y_std).log_prob(y.squeeze())))
        
def save(save_root, history=None, epoch=None, Enc=None, Latent_func=None, Dec=None, save=True):
    if save:
        if history != None:
            if np.argmin([h['nll'] for h in history.epoch_history]) == len(history.epoch_history)-1:
                if Enc!= None:
                    ENC_BEST_PATH = save_root + 'Enc_Best.pt'
                    torch.save(Enc.state_dict(), ENC_BEST_PATH)
                if Latent_func != None:
                    ODE_BEST_PATH = save_root + 'ODE_Best.pt'
                    torch.save(Latent_func.state_dict(), ODE_BEST_PATH)
                if Dec != None:
                    DEC_BEST_PATH = save_root + 'Dec_Best.pt'
                    torch.save(Dec.state_dict(), DEC_BEST_PATH)

        if np.mod(epoch, 500) == 0:
            if Enc!=None:
                ENC_CHKPT_PATH = save_root + 'Enc_chkpt_' + str(epoch) + '.pt'
                torch.save(Enc.state_dict(), ENC_CHKPT_PATH)
            if Latent_func!=None:
                ODE_CHKPT_PATH = save_root + 'ODE_chkpt_' + str(epoch) + '.pt'
                torch.save(Latent_func.state_dict(), ODE_CHKPT_PATH)
            if Dec != None:
                DEC_CHKPT_PATH = save_root + 'Dec_chkpt_' + str(epoch) + '.pt'
                torch.save(Dec.state_dict(), DEC_CHKPT_PATH)

    else:
        if Enc!= None:
            ENC_BEST_PATH = save_root + 'Enc_Best.pt'
            Enc.load_state_dict(torch.load(ENC_BEST_PATH))
        if Latent_func != None:
            ODE_BEST_PATH = save_root + 'ODE_Best.pt'
            Latent_func.load_state_dict(torch.load(ODE_BEST_PATH))
        if Dec != None:
            DEC_BEST_PATH = save_root + 'Dec_Best.pt'
            Dec.load_state_dict(torch.load(DEC_BEST_PATH))
        return Enc, Latent_func, Dec