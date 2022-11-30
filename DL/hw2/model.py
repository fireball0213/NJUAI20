import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    This file contains main arch of models, including:
    - AutoEncoder
    - Variational AutoEncoder
    - Conditional Variational AutoEncoder
    
"""


class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size, is_dist=False, **kwargs) -> None:
        super(Encoder, self).__init__()
        self.mu = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size),)
        if is_dist:  # 如果需要encoder返回的是均值与标准差, 那么我们需要额外引入一次计算标准差的layer
            self.sigma = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size),)

    def forward(self, xs):
        # 实现编码器的forward过程 (5/100), 注意 is_dist 的不同取值意味着我们需要不同输出的encoder
        ...
        return


class Decoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size, decode_type="AE", **kwargs) -> None:
        super(Decoder, self).__init__()
        if decode_type == "AE":
            self.decoder = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, x_dim),)
        elif decode_type == "VAE":
            self.decoder = ...  # 实现VAE的decoder (5/100)
        elif decode_type == "CVAE":
            self.decoder = ...  # 实现CVAE的decoder (5/100)
        else:
            raise NotImplementedError

    def forward(self, zs, **otherinputs):
        # 实现decoder的decode部分, 注意对不同的decode_type的处理与对**otherinputs的解析 (10/100)
        return ...


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        # 实现AE的forward过程(5/100)
        ...
        return


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs):
        mu, sigma = self.encoder(xs)
        # 实现VAE的forward过程(10/100)
        ...
        return


class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(ConditionalVariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs, ys):
        mu, sigma = self.encoder(xs, ys)
        # 实现 CVAE的forward过程(15/100)
        ...
        return
