import numpy as np
import tqdm
import torch
from itertools import chain
from torchdiffeq import odeint
import lib.train_functions as train_functions
from torch.distributions import Normal
from lib.in_development.models_bayes import Dense_Variational
import lib.models as models
import lib.Metrics as Metrics

def print_rounded_dict(dictionary, decimals=3):
    formatted_dict = {key: round(value, decimals) if isinstance(value, (int, float)) else value for key, value in dictionary.items()}
    print(formatted_dict)
    
def evaluate(model, x_test, y_test, t, scaler, n_samples=128):

    y_pred = model(x_test, t, n_samples=n_samples)
    y_pr = y_pred.detach().numpy() * scaler.values[np.newaxis, np.newaxis, np.newaxis, :]
    y_te = y_test.detach().numpy() * scaler.values[np.newaxis, np.newaxis, :]

    pred_mean = y_pr.mean(1)
    pred_std = y_pr.std(1)
    nlls = [Metrics.nll(y_te[:, g, :], pred_mean[:, g, :], pred_std[:, g, :]) for g in range(57)]
    return {'forecast_nll': np.mean(nlls[-28:]), 'all_nll':np.mean(nlls)}

class VAE:
    def __init__(self, enc, ode, dec, n_qs, latent_dim, 
                 n_regions=1,
                 ode_type='Fp', 
                 len_tr=130, 
                 file_prefix = None,
                 chkpt_prefix = None,
                 prior_params = {'means':[0.8, 0.55],
                                 'stds':[0.2, 0.2]},
                 device='cpu', 
                 ode_params = {},
                 enc_params = {},
                 dec_params = {},
                 kl_w =1,
                 ode_kl_w = 1,
                 uncertainty = True,
                 dtype=torch.float32):
        self.kl_w = kl_w
        self.ode_kl_w = ode_kl_w
        
        self.dtype = dtype
        self.device = device
        self.tr_step = 0
        self.n_regions = n_regions
        self.len_tr = len_tr
        self.ld_ode = latent_dim
        self.uncertainty = uncertainty

        self.ode = ode(n_regions, latent_dim = self.ld_ode, **ode_params)

        self.file_prefix=file_prefix
        self.chkpt_prefix=chkpt_prefix

        self.ode_type = ode_type
        if hasattr(self.ode, 'ode_type'):
            self.ode_type = self.ode.ode_type
        
        if ode_type == 'Fa':
            self.ld_enc = latent_dim
        else:
            self.ld_enc = latent_dim-1
            self.ld_dec = 3 
        
        self.enc = enc(n_regions, 
                       n_qs=n_qs, 
                       latent_dim = self.ld_enc,
                       device=device, 
                       dtype=dtype, 
                       uncertainty = uncertainty,
                       **enc_params)
        
        self.dec = dec(n_regions, 
                        latent_dim=self.ld_dec, 
                        input_dim=1,
                        **dec_params)
        
        self.anneal_params = {'anneal':True,
                           'reset_pos':10000,
                           'split':0.5, 
                           'lower':0.0, 
                           'upper':1.0, 
                           'type':'cosine'
                           }
        
        self.batch_grad_norms = []
        self.prior_params = prior_params
        self.started = False

    def update_priors(self, new_std = 0.1):
        for net in [self.ode.aug_net, self.ode.Fp_net]:
            try:
                for name, layer in net.named_modules():
                    if isinstance(layer, Dense_Variational):
                        layer.prior_std = new_std    
            except:
                pass

    def setup_training(self, lr = 1e-3):
        self.optimizer = torch.optim.Adam(chain(self.enc.parameters(), 
                                                self.ode.parameters(), 
                                                self.dec.parameters()), lr=lr)
        self._history = train_functions.history()
        
    def __call__(self, x, t, n_samples=32, training=False):
        batch_size = x.shape[0]
        eps = torch.randn(n_samples, batch_size, self.n_regions, self.ld_enc, dtype=self.dtype, device=self.device)
        self.ode.clear_tracking()
        
        if training:
            self.optimizer.zero_grad()

        step_size = t[1] - t[0]
        with torch.set_grad_enabled(training):
            if self.uncertainty:
                self.mean, self.std = self.enc(x)
                z = models.reparam(eps, self.std, self.mean, n_samples, batch_size, uncertainty=True) + 1e-5
            else:
                n_samples = 1
                self.mean = self.enc(x)
                z = models.reparam(eps, None, self.mean, n_samples, batch_size, uncertainty=False) + 1e-5
                z = z.unsqueeze(1)
                
            self.latent = odeint(self.ode, z, t, method='rk4', options=dict(step_size=step_size))
            y_pred = self.dec(self.latent[..., :3]).reshape((-1, n_samples, batch_size, self.n_regions)).permute(2, 1, 0, 3)

        return y_pred
       
    def calc_loss(self, y_pred, y_true, losses):
        loss = torch.tensor(0.0, requires_grad=True)

        batch_data = [0]
        batch_names = ['loss']

        if losses.get('anneal', True):
            self.tr_step = self.tr_step + 1
            self.kl_w = train_functions.KL_annealing(self.tr_step, self.anneal_params)
            batch_data.append(round(self.kl_w, 3))
            batch_names.append('kl_w')

        if losses.get('mse', True):
            mse = torch.mean(torch.square(y_pred - y_true.unsqueeze(1)))
            loss = loss + mse
            batch_data.append(round(mse.cpu().item(), 3))
            batch_names.append('mse')

        if losses.get('nll', True):
            nll = train_functions.nll_loss(y_pred, y_true)
            loss = loss + nll
            batch_data.append(round(nll.cpu().item(), 3))
            batch_names.append('nll')
                               
        if losses.get('kl_z', True):
            kl_z = self.kl_w * train_functions.kl_divergence(models.make_prior(self.mean, latent_dim=self.ld_ode, device=self.device), Normal(self.mean, self.std)).sum(-1).mean() / self.len_tr
            loss = loss+ kl_z
            batch_data.append(round(kl_z.cpu().item(), 3))
            batch_names.append('kl_latent')

        if losses.get('kl_p', True):
            kl_p = train_functions.get_kl_params(1, self.ode.posterior(), means=self.prior_params['means'],
                                                stds=self.prior_params['stds'], limit=1e6, device=self.device)
            loss = loss+ kl_p
            batch_data.append(round(kl_p.cpu().item(), 3))
            batch_names.append('kl_params')
        
        if losses.get('Fa_norm', True):
            norm = torch.norm(torch.stack(self.ode.tracker))
            loss = loss+ norm
            batch_data.append(round(norm.cpu().item(), 3))
            batch_names.append('Fa_norm')
        
        if losses.get('reg_loss', True):
            reg_loss = 0.1 * train_functions.latent_init_loss(self.latent[..., :3])
            loss = loss+ reg_loss
            batch_data.append(round(reg_loss.cpu().item(), 3))
            batch_names.append('reg_loss')

        if self.ode.uncertainty == 'bayes':
            ode_kl = self.ode_kl_w * self.ode.get_kl()
            loss = loss+ ode_kl
            batch_data.append(round(ode_kl.cpu().item(), 3))
            batch_names.append('ode_kl')
                                             
        batch_data[0] = round(loss.cpu().item(), 3)
        return loss, batch_data, batch_names

    def train_step(self, x, y, t, epoch, losses, eval_pts, grad_lim = 300, n_samples=32, track_norms = False, norm_file = 'grad_norms.txt'):
        y_pred = self(x, t[eval_pts], n_samples=n_samples, training = True)
        loss, batch_data, batch_names = self.calc_loss(y_pred, y[:, eval_pts, :], losses=losses)
        loss.backward()

        grad_norm = torch.norm(torch.cat([p.grad.data.view(-1) for p in chain(self.enc.parameters(), self.ode.parameters(), self.dec.parameters())]), 2).item()
        self.batch_grad_norms.append(grad_norm)

        if grad_norm < grad_lim or self.skip_count >= 4 or epoch <= 3:
            self.optimizer.step()
            self.skip_count = 0
        else:
            self.skip_count = self.skip_count + 1

        batch_data.append(round(grad_norm, 1))
        batch_names.append('grad_norm')

        if track_norms:
            if not self.started:
                with open(norm_file, 'w') as file:
                    file.write('')
                self.started = True
            self.norms.append(round(grad_norm, 1))
        return (batch_data, batch_names)
    
    def pre_train(self, train_loader, epochs=3, lr=1e-3, disable = False): 
        optimizer = torch.optim.Adam(self.enc.parameters(), lr=lr)
        for epoch in range(1,1+epochs):
            kls = []
            
            pbar = tqdm.tqdm(train_loader, desc="Training", leave=True, disable = disable)
            for batch in pbar:
                x_batch, y_batch = batch
                optimizer.zero_grad()
                
                mean, std = self.enc(x_batch)        
                
                kl_z = train_functions.kl_divergence(models.make_prior(mean, latent_dim=self.ld_ode, device=self.device), 
                                                       Normal(mean, std)).sum(-1).mean() / self.len_tr
                
                kl_z.backward()
                optimizer.step()
                
                kls.append(kl_z.cpu().detach().numpy())
                pbar.set_postfix({'Epoch':epoch, 'KL_z':np.mean(kls)})  
            if disable:
                print(f"{'Epoch':<8}: {round(epoch, 3):.3f}, {'KL_z':<8}: {round(np.mean(kls), 3):.3f}")

    def train(self, train_loader, t, epochs, losses, eval_pts, grad_lim=300, n_samples=32, checkpoint=False, track_norms = False, norm_file = 'grad_norms.txt', disable=False, validate = None): 
        self.best_loss = 1e9 
        self.skip_count = 0

        for epoch in range(epochs):
            pbar = tqdm.tqdm(train_loader, desc="Training " + str(epoch+1), leave=True, disable=disable)    
            self.norms = []
            
            for x, y in pbar:
                loss_data = self.train_step(x, y, t, epoch, losses, eval_pts, grad_lim=grad_lim, n_samples=n_samples, track_norms=track_norms, norm_file=norm_file)
                self._history.batch(loss_data[0], loss_data[1])
                pbar.set_postfix(self._history.epoch())
                
            self._history.reset()
            
            if validate is not None:
                y_pred = self(validate['x_test'], validate['t'], n_samples=validate['n_samples'], training = False)
            
                y_pr = y_pred.detach().numpy() * validate['scaler'].values[np.newaxis, np.newaxis, np.newaxis, :]
                y_te = validate['y_test'].detach().numpy() * validate['scaler'].values[np.newaxis, np.newaxis, :]
    
                pred_mean = y_pr.mean(1)
                pred_std = y_pr.std(1)
                nlls = [Metrics.nll(y_te[:, g, :], pred_mean[:, g, :], pred_std[:, g, :]) for g in range(len(t))]
                
                self._history.epoch_history[-1]['forecast_nll'] =  np.mean(nlls[-28:])
                self._history.epoch_history[-1]['all_nll'] =  np.mean(nlls)
                
            if disable:
                print(epoch, end = ' ')
                print_rounded_dict(self._history.epoch_history[-1], decimals=3)
                
            with open(norm_file, 'a') as file:
                file.write(','.join(map(str, self.norms)) + '\n')

            if checkpoint:
                self.checkpoint()
                
    def checkpoint(self):
        if self.chkpt_prefix == None:
            self.chkpt_prefix = self.file_prefix

        if self._history.epoch_history[-1]['loss'] < self.best_loss:
            self.best_loss = self._history.epoch_history[-1]['loss']
                
            enc_file = f'{self.chkpt_prefix}chkpt_enc.pth'
            ode_file = f'{self.chkpt_prefix}chkpt_ode.pth'
            dec_file = f'{self.chkpt_prefix}chkpt_dec.pth'

            torch.save(self.enc.state_dict(), enc_file)
            torch.save(self.dec.state_dict(), dec_file)
            torch.save(self.ode.state_dict(), ode_file)

    def save(self): 
        enc_file = f'{self.file_prefix}enc.pth'
        ode_file = f'{self.file_prefix}ode.pth'
        dec_file = f'{self.file_prefix}dec.pth'

        torch.save(self.enc.state_dict(), enc_file)
        torch.save(self.dec.state_dict(), dec_file)
        torch.save(self.ode.state_dict(), ode_file)

    def load(self, checkpoint = False, file_prefix=None):    
        if self.chkpt_prefix == None:
            self.chkpt_prefix = self.file_prefix 
        if file_prefix == None:
            file_prefix = self.file_prefix
    
        if checkpoint:
            enc_file = f'{self.chkpt_prefix}chkpt_enc.pth'
            ode_file = f'{self.chkpt_prefix}chkpt_ode.pth'
            dec_file = f'{self.chkpt_prefix}chkpt_dec.pth'
        else:
            enc_file = f'{file_prefix}enc.pth'
            ode_file = f'{file_prefix}ode.pth'
            dec_file = f'{file_prefix}dec.pth'

        self.enc.load_state_dict(torch.load(enc_file), strict=False)
        self.dec.load_state_dict(torch.load(dec_file), strict=False)
        self.ode.load_state_dict(torch.load(ode_file), strict=False)


