import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    This file contains main arch of models, including:
    - AutoEncoder
    - Variational AutoEncoder
    - Conditional Variational AutoEncoder
    
"""
def idx2onehot(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)
    return onehot

class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size, is_dist=False, conditional=False, **kwargs) -> None:
        super(Encoder, self).__init__()
        self.mu = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size), )
        self.is_dist=is_dist
        self.conditional=conditional
        if self.is_dist:  # 如果需要encoder返回的是均值与标准差, 那么我们需要额外引入一次计算标准差的layer
            self.sigma = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size), )

    def forward(self, xs, c=None):
        # 实现编码器的forward过程 (5/100), 注意 is_dist 的不同取值意味着我们需要不同输出的encoder
        enc_outputs = self.mu(xs)
        if self.is_dist:
            if self.conditional:
                c = idx2onehot(c, n=10)
                xs= torch.cat((xs, c), dim=-1)
            mu = self.sigma(xs)#使用不同网络
            sigma = self.sigma(xs)
            return mu,sigma
        return enc_outputs  # (128,1,10)


class Decoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size, decode_type="AE", conditional=False, **kwargs) -> None:
        super(Decoder, self).__init__()
        self.conditional = conditional
        if decode_type == "AE":
            self.decoder = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(),
                                         nn.Linear(hidden_size, x_dim), )
        elif decode_type == "VAE":
            self.decoder = ...  # 实现VAE的decoder (5/100)
            self.decoder = nn.Sequential(nn.Linear(latent_size, 50), nn.ReLU(),
                                         nn.Linear(50, hidden_size), nn.ReLU(),
                                         nn.Linear(hidden_size, 400), nn.ReLU(),
                                         nn.Linear(400, x_dim)
                                         )
        elif decode_type == "CVAE":
            self.decoder = ...  # 实现CVAE的decoder (5/100)
            self.decoder = nn.Sequential(nn.Linear(latent_size, 50), nn.ReLU(),
                                         nn.Linear(50, hidden_size), nn.ReLU(),
                                         nn.Linear(hidden_size, 400), nn.Sigmoid(),
                                         nn.Linear(400, x_dim)
                                         )
        else:
            raise NotImplementedError

    def forward(self, zs,c=None, **otherinputs):
        # 实现decoder的decode部分, 注意对不同的decode_type的处理与对**otherinputs的解析 (10/100)
        if self.conditional:
            c = idx2onehot(c, n=10)
            zs = torch.cat((zs, c), dim=-1)
        dec_outputs = self.decoder(zs)
        return dec_outputs  # (128,1,784)


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.batch_size = kwargs['batch_size']

    def forward(self, x):
        z = self.encoder(x)
        # 实现AE的forward过程(5/100)
        x_ = self.decoder(z)
        return z, x_


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs):
        # 实现VAE的forward过程(10/100)
        mu, log_var = self.encoder(xs)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        x_ = self.decoder(z)
        return x_, mu,log_var, z
    def inference(self, z):
        x_ = self.decoder(z)
        return x_


class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(ConditionalVariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs, ys):
        # 实现 CVAE的forward过程(15/100)
        mu, log_var = self.encoder(xs,ys)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        x_ = self.decoder(z,ys)

        return x_, mu, log_var, z

    def inference(self, z,y):
        x_ = self.decoder(z,y)
        return x_