import argparse
import os

import torch
import torch.nn as nn

from data import *
from model import *


class Loss(nn.Module):
    def __init__(self, decode_type="AE", **kwargs) -> None:
        super().__init__()
        self.decode_type = decode_type

    def recon_loss(self, x, x_):
        # 实现重构损失函数 (5/100)
        return

    def kl_div(self, p, q):
        # 实现kl散度的损失 (5/100)
        return

    def forward(self, x, x_, **otherinputs):
        # 实现loss的计算, 注意处理 **otherinputs 的解析, 以及不同的decode_type对应于不同的loss (10/100)
        return


def train(model, loader, loss, optimizer, epoch_num, type):
    for epoch in range(epoch_num):
        for i, (x, y) in enumerate(loader):
            # 训练过程的补全 (20/100) 注意考虑不同类型的AE应该有所区别
            ...

        with torch.no_grad():
            if not os.path.exits(f"./results/{type}/epoch_{epoch}"):
                os.mkdir(f"./results/{type}/epoch_{epoch}")
            # 保存一些重构出来的图像用于(写报告)进一步研究 (5/100)

    return


def main(args):
    encoder_args = {
        "x_dim": args.x_dim,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "is_dist": True if args.type in ["VAE", "CVAE"] else False,
    }
    encoder_args = {
        "x_dim": args.x_dim,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "decode_type": args.type,
    }
    encoder = Encoder(**encoder_args)
    decoder = Decoder(**encoder_args)
    ae = {"AE": AutoEncoder, "VAE": VariationalAutoEncoder, "CVAE": ConditionalVariationalAutoEncoder}

    auto_encoder = ae[args.type](encoder, decoder)
    # 挑选你喜欢的优化器 :)
    optimizer = torch.optim.SGD(auto_encoder.parameter(), lr=0.01)
    train_loader = get_data(train=True, batch_size=args.batch_size)
    #test loader 可以用于单独的验证模型的训练结果, 这部分可以考虑复用train函数, 或是把train函数中evaluate的部分单独抽出来...
    test_loader = get_data(train=False, batch_size=args.batch_size)
    loss = Loss(args.type)
    train(model=auto_encoder, loss=loss, loader=train_loader, optimizer=optimizer, epoch_num=args.epoch_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="AE", choices=["AE", "VAE", "CVAE"])
    parser.add_argument("--x_dim", default=784, type=int)
    parser.add_argument("--latent_size", default=10, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epoch_num", default=128, type=int)
