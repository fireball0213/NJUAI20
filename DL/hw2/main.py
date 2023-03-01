import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from data import *
from model import *


class Loss(nn.Module):
    def __init__(self, decode_type="AE", **kwargs) -> None:
        super().__init__()
        self.decode_type = decode_type
    def recon_loss(self, x, x_):
        # 实现重构损失函数 (5/100)
        # L1
        loss = nn.L1Loss()  # 必须导这么一手
        recon = loss(x, x_)
        #BCE
        #recon = F.binary_cross_entropy(F.sigmoid(x_),F.sigmoid(x), reduction='sum')
        return recon

    def kl_div(self, mu,logvar):
        # 实现kl散度的损失 (5/100)
        kl_mean = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_mean

    def forward(self, x, x_,mu,logvar,epoch,epoch_num, **otherinputs):
        # 实现loss的计算, 注意处理 **otherinputs 的解析, 以及不同的decode_type对应于不同的loss (10/100)
        recon=Loss.recon_loss(self,x,x_)
        if self.decode_type=="AE":
            total_loss=recon
        elif self.decode_type=="VAE" or self.decode_type=="CVAE":
            kl = Loss.kl_div(self,mu,logvar)
            annealing=epoch/epoch_num
            annealing*=0.003
            total_loss=recon+kl*annealing
        else:
            raise NotImplementedError
        return total_loss

def train_epoch(model, train_loader, loss, optimizer,epoch, type,epoch_num):
    # 训练过程
    model.train()
    train_loss = 0
    for i, (x, y) in enumerate(train_loader):
        # 训练过程的补全 (20/100) 注意考虑不同类型的AE应该有所区别
        x = x.reshape(128, 1, -1)  # (128,1,784)28*28需要展平
        optimizer.zero_grad()
        if type=="AE":
            z, x_ = model(x)
            loss_num = loss(x, x_)
        elif  type=="VAE":
            x_, mu, log_var, z=model(x)
            loss_num=loss(x, x_,mu,log_var,epoch,epoch_num)
        elif  type=="CVAE":
            x_, mu, log_var, z=model(x,y)
            loss_num=loss(x, x_,mu,log_var,epoch,epoch_num)
        else:
            raise NotImplementedError
        train_loss += loss_num
        loss_num.backward()
        optimizer.step()

        if i % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(x), len(train_loader.dataset),
                       100. * i / len(train_loader),
                loss_num.item())  # / len(x)
            )
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch,
        train_loss * 128 / len(train_loader.dataset)))
    return


def test_epoch(model, test_loader, loss, epoch, type,epoch_num):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.reshape(128, 1, -1)

            if type == "AE":
                z, x_ = model(x)
                loss_num = loss(x, x_)
                test_loss += loss_num.item()

                # 保存一些重构出来的图像用于(写报告)进一步研究 (5/100)
                with torch.no_grad():
                    if i%30==0:
                        # 保存原始数据部分
                        if not os.path.exists(f"datas/{type}"):
                            os.makedirs(f"datas/{type}")
                        save_x = x.reshape(-1,1, 28, 28)[:16]
                        save_x = make_grid(save_x,8,0)
                        save_image(save_x,os.path.join(f"datas/{type}/batch_{i}.png"))

                        # 保存训练结果部分
                        if not os.path.exists(f"results/{type}/epoch_{epoch}"):
                            os.makedirs(f"results/{type}/epoch_{epoch}")
                        save_x_ = x_.reshape(-1, 1, 28, 28)[:16]
                        save_x_ = make_grid(save_x_, 8, 0)
                        save_image(save_x_, os.path.join(f"results/{type}/epoch_{epoch}/batch_{i}.png"))

            elif type == "VAE":
                x_, mu, log_var, z = model(x)
                loss_num = loss(x, x_, mu, log_var, epoch, epoch_num)
                test_loss += loss_num.item()
                z = torch.randn(128,1,10)
                sample = model.inference(z)
                sample = sample.reshape(-1, 1, 28, 28)[:32]
                #print(z[0],sample[0])
                # 保存一些重构出来的图像用于(写报告)进一步研究 (5/100)
                if not os.path.exists(f"results/{type}"):
                    os.makedirs(f"results/{type}")
                save_image(
                    sample,
                    os.path.join(f"results/{type}/epoch_{epoch}.png")
                )
            elif type == "CVAE":
                x_, mu, log_var, z = model(x,y)
                loss_num = loss(x, x_, mu, log_var, epoch, epoch_num)
                test_loss += loss_num.item()
                z = torch.randn(128,1,10)
                sample = model.inference(z,y)
                sample = sample.reshape(-1, 1, 28, 28)[:32]
                #print(z[0],sample[0])
                # 保存一些重构出来的图像用于(写报告)进一步研究 (5/100)
                if not os.path.exists(f"results/{type}"):
                    os.makedirs(f"results/{type}")
                save_image(
                    sample,
                    os.path.join(f"results/{type}/epoch_{epoch}.png")
                )

            else:
                raise NotImplementedError

    test_loss*=128
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main(args):
    encoder_args = {
        "x_dim": args.x_dim,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "decode_type": args.type,
        "is_dist": True if args.type in ["VAE", "CVAE"] else False,
    }
    encoder = Encoder(**encoder_args)
    decoder = Decoder(**encoder_args)
    ae = {"AE": AutoEncoder, "VAE": VariationalAutoEncoder, "CVAE": ConditionalVariationalAutoEncoder}

    auto_encoder = ae[args.type](encoder, decoder)

    # 挑选你喜欢的优化器 :)
    #optimizer = torch.optim.SGD(auto_encoder.parameters(), lr=0.01)
    optimizer =torch.optim.Adam(auto_encoder.parameters(), lr=0.001)
    train_loader = get_data(train=True, batch_size=args.batch_size)
    # test loader 可以用于单独的验证模型的训练结果, 这部分可以考虑复用train函数, 或是把train函数中evaluate的部分单独抽出来...
    test_loader = get_data(train=False, batch_size=args.batch_size)
    loss = Loss(args.type)
    # train(model=auto_encoder, loss=loss, train_loader=train_loader, test_loader=test_loader,
    #       optimizer=optimizer, epoch_num=args.epoch_num,type=args.type)
    for epoch in range(args.epoch_num):
        print("\n epoch:", epoch)
        train_epoch(model=auto_encoder, loss=loss, train_loader=train_loader,
                    optimizer=optimizer, epoch=epoch, type=args.type,epoch_num=args.epoch_num)
        test_epoch(model=auto_encoder, test_loader=test_loader, loss=loss,
                   epoch=epoch, type=args.type, epoch_num=args.epoch_num)



if __name__ == "__main__":
    matplotlib.use('TkAgg')
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="CVAE", choices=["AE", "VAE", "CVAE"])
    parser.add_argument("--x_dim", default=784, type=int)  # 28 x 28 的像素展开为一个一维的行向量，每行代表一个图片
    parser.add_argument("--latent_size", default=10, type=int)  # 输出层大小，即服从高斯分布的隐含变量的维度。
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epoch_num", default=128, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
    # parser.set_defaults(type="CVAE")
    # args = parser.parse_args()
    # main(args)
