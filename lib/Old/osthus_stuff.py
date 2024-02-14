import torch
from torch.distributions import Normal

def make_ics(x_0, I_0 = 0.9, latent = 8, std = 0.002, n = 32, dtype=torch.float32):
    s = Normal(I_0, std).sample((n,)).unsqueeze(-1)
    i = Normal(x_0, std).sample((n,)).unsqueeze(-1)
    r = 1-s - torch.abs(i)
    extra = Normal(torch.zeros(latent-3), torch.ones(latent-3)).sample((n,))
    return torch.concat([s, i, r, extra], -1)

def reparam(mean, std, z):
    IC = torch.abs(mean + z*std)
    IC = torch.concat([IC, (1 - IC.sum(-1).unsqueeze(-1))], -1).squeeze()
    return IC