import os
import numpy as np
import pandas as pd
import datetime as dt
import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
from scipy.stats import norm
import torch
from itertools import chain
from torch.utils.data import TensorDataset, DataLoader

from lib.encoder_back_gru import Encoder_Back_GRU
import lib.Metrics as Metrics
from lib.regional_data_builder import DataConstructor, convert_to_torch
# from lib.encoders import 
from torch import nn
import lib.encoders as encoders
from torchdiffeq import odeint
import lib.train_functions as train_functions
import lib.utils as utils
from lib.UONN import FaFp
from lib.models import Decoder, Fa, Fp
from torch.distributions import Categorical, Normal
from lib.VAE import VAE
from filelock import FileLock

torch.set_num_threads(1)

if __name__ == '__main__':
    batch_size = 32
    window_size = 28
    n_regions = 49
    n_qs = 9
    latent_dim = 8
    lr = 1e-3
    dtype = torch.float32
    gamma = 28
    n_samples = 64

    mydict =   {'Full_FaFp':['UONN', {'nll':True, 'kl_z':True, 'kl_p':True,  'Fa_norm':True,  'reg_loss':True,  'anneal':True}],
                'Half_FaFp':['UONN', {'nll':True, 'kl_z':True, 'kl_p':False, 'Fa_norm':True,  'reg_loss':False, 'anneal':True}],
                'Full_Fp':  ['CONN', {'nll':True, 'kl_z':True, 'kl_p':True,  'Fa_norm':False, 'reg_loss':True,  'anneal':True}],
                'Full_Fa':  ['CONN', {'nll':True, 'kl_z':True, 'kl_p':False, 'Fa_norm':False, 'reg_loss':False, 'anneal':True}],
                'Fa':       ['SONN', {'nll':True, 'kl_z':True, 'kl_p':False, 'Fa_norm':False, 'reg_loss':False, 'anneal':True}],
               }

    t = torch.arange(window_size + gamma + 1, dtype=dtype)/7
    eval_pts = np.arange(0,t.shape[-1], 7)
    epochs = 250
    q_sizes = [256, 128]
    ff_sizes = [128, 64]
    n_qs = 5
    latent_dim = 6
    epochs = 250
    region = 'state'
    started_file_path = "HHS_started.txt"
    ode_name = 'UONN'
    for num in range(10, 16):
        for test_season in [2015, 2016, 2017, 2018]:
            for key in list(mydict.keys()):
                file_prefix = f'weights/{ode_name}_{key}/{region}_{num}_{test_season}_'
                run = False
                with FileLock(started_file_path + ".lock"):
                    with open(started_file_path, 'r') as file:
                        content = file.read().splitlines()

                    if file_prefix not in content:
                        run = True
                
                if run:
                    if not os.path.exists(f'weights/{ode_name}_{key}'):
                        os.mkdir(f'weights/{ode_name}_{key}')

                    with FileLock(started_file_path + ".lock"):
                        with open(started_file_path, 'a') as file:
                            file.write(file_prefix + '\n')

                    print(file_prefix)
                    ode_name = mydict[key][0]
                    losses = mydict[key][1]
                    ode = {'CONN':Fp, 'UONN':FaFp, 'SONN':Fa}[ode_name]

                    _data = DataConstructor(test_season=test_season, region = region, window_size=window_size, n_queries=n_qs, gamma=gamma)
                    x_train, y_train, x_test, y_test, scaler = _data(run_backward=True, no_qs_in_output=True)
                    train_loader, x_test, y_test = convert_to_torch(x_train, y_train, x_test, y_test, batch_size=32, shuffle=True, dtype=dtype)  

                    model = VAE(Encoder_Back_GRU, ode, Decoder, n_qs, latent_dim, n_regions, q_sizes=q_sizes, ff_sizes=ff_sizes, file_prefix=file_prefix)
                    model.setup_training(lr=lr)

                    model.pre_train(train_loader, 3)
                    model.train(train_loader, t, epochs, losses, eval_pts, n_samples = n_samples, grad_lim=5000, checkpoint=True, track_norms=True, norm_file=f'{file_prefix}norms.txt')
                    model.save()







# import os
# import numpy as np
# import pandas as pd
# import datetime as dt
# import tqdm
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from scipy import interpolate
# from scipy.stats import norm
# import torch
# from itertools import chain
# from torch.utils.data import TensorDataset, DataLoader

# from lib.encoder_back_gru import Encoder_Back_GRU
# import lib.Metrics as Metrics
# from lib.hhs_data_builder import DataConstructor, convert_to_torch
# # from lib.encoders import 
# from torch import nn
# import lib.encoders as encoders
# from torchdiffeq import odeint
# import lib.train_functions as train_functions
# import lib.utils as utils
# from lib.UONN import FaFp
# from lib.models import Decoder, Fa, Fp
# from torch.distributions import Categorical, Normal
# from lib.VAE import VAE
# from filelock import FileLock

# torch.set_num_threads(1)

# if __name__ == '__main__':
#     batch_size = 32
#     window_size = 28
#     n_regions = 10
#     n_qs = 9
#     latent_dim = 8
#     lr = 1e-3
#     dtype = torch.float32
#     gamma = 28
#     region = 'hhs'

#     t = torch.arange(window_size + gamma + 1, dtype=dtype)/7
#     eval_pts = np.arange(0,t.shape[-1], 7)
#     epochs = 400
#     started_file_path = "HHS_started.txt"
#     ode_name = 'UONN'
#     for num in range(10, 16):
#         for test_season in [2015, 2016, 2017, 2018]:
#             for ode_name in ['UONN']:
#                 file_prefix = f'weights/{ode_name}/{region}_{num}_{test_season}_'
#                 run = False
                
#                 with FileLock(started_file_path + ".lock"):
#                     with open(started_file_path, 'r') as file:
#                         content = file.read().splitlines()

#                     if file_prefix not in content:
#                         run = True
                
#                 if run:
#                     with FileLock(started_file_path + ".lock"):
#                         with open(started_file_path, 'a') as file:
#                             file.write(file_prefix + '\n')
                            
#                     print(ode_name, test_season, num)

#                     ode = {'CONN':Fp, 'UONN':FaFp, 'SONN':Fa}[ode_name]

#                     _data = DataConstructor(test_season=test_season, region = region, window_size=window_size, n_queries=n_qs, gamma=gamma)
#                     x_train, y_train, x_test, y_test, scaler = _data(run_backward=True, no_qs_in_output=True)
#                     train_loader, x_test, y_test = convert_to_torch(x_train, y_train, x_test, y_test, batch_size=32, shuffle=True, dtype=dtype)  

#                     model = VAE(Encoder_Back_GRU, ode, Decoder, n_qs, latent_dim, n_regions, file_prefix=file_prefix)
#                     model.setup_training(lr=lr)
#                     model.load(file_prefix = f'weights/{"CONN"}/{region}_{3}_{2015}_')

#                     model.ode.Fa_w = 0.0
#                     model.train(train_loader, t, 10, eval_pts, grad_lim=650, n_samples = 32, checkpoint=True, track_norms=True, norm_file=f'{file_prefix}norms.txt')
#                     model.ode_type = 'FaFp'
#                     for run in range(10):
#                         model.ode.Fa_w = 0.01 * run
#                         model.train(train_loader, t, 1, eval_pts, grad_lim=650, n_samples = 32, checkpoint=True, track_norms=True, norm_file=f'{file_prefix}norms.txt')
#                     model.train(train_loader, t, 80, eval_pts, grad_lim=650, n_samples = 32, checkpoint=True, track_norms=True, norm_file=f'{file_prefix}norms.txt')

                    
                    

#                     # model.pre_train(train_loader, 3)
#                     # model.train(train_loader, t, epochs, eval_pts, grad_lim=50, checkpoint=True, track_norms=True, norm_file=f'{file_prefix}norms.txt')
#                     model.save()