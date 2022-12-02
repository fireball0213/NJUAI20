import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from data import *
from model import *


class Loss(nn.Module):
    def __init__(self, decode_type="AE", **kwargs) -> None:
        super().__init__()
        self.decode_type = decode_type
        #self.epoch=kwargs['epoch']
    def recon_loss(self, x, x_):
        # 实现重构损失函数 (5/100)
        #BCE
        #recon = F.binary_cross_entropy(F.sigmoid(x_),F.sigmoid(x), reduction='mean')
        #L1
        loss = nn.L1Loss()#必须导这么一手
        recon=loss(x,x_)
        return recon

    def kl_div(self, p, q):  # p预测，q目标(y)
        # 实现kl散度的损失 (5/100)
        log_p = F.log_softmax(p, dim=-1)
        softmax_q = F.softmax(q, dim=-1)
        assert (softmax_q[-1].sum() == 1)
        kl_mean = F.kl_div(log_p, softmax_q, reduction='mean')
        assert (kl_mean >= 0)
        return kl_mean

    def forward(self, x, x_, **otherinputs):
        # 实现loss的计算, 注意处理 **otherinputs 的解析, 以及不同的decode_type对应于不同的loss (10/100)
        #kl = Loss.kl_div(self,x, x_)
        recon=Loss.recon_loss(self,x,x_)

        if self.decode_type=="AE":
            total_loss=recon
        # elif self.decode_type=="VAE":
        #     total_loss=recon+kl#TODO:*annealing
        else:
            raise NotImplementedError
        return total_loss

def train_epoch(model, train_loader, loss, optimizer,epoch, type):
    # 训练过程
    model.train()
    train_loss = 0
    for i, (x, y) in enumerate(train_loader):
        # 训练过程的补全 (20/100) 注意考虑不同类型的AE应该有所区别
        x = x.reshape(128, 1, -1)  # (128,1,784)28*28需要展平
        optimizer.zero_grad()
        z, x_ = model(x)
        x = x.requires_grad_(True)
        x_ = x_.requires_grad_(True)
        loss_num = loss(x, x_)
        train_loss += loss_num
        loss_num.backward()
        optimizer.step()
        if i % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(x), len(train_loader.dataset),
                       100. * i / len(train_loader),
                loss_num.item())  # / len(x)
            )
            # 保存一些重构出来的图像用于(写报告)进一步研究 (5/100)
            # 保存原始数据部分
            if epoch==0:
                save_x = x.reshape(128, 28, 28)
                if not os.path.exists(f"data_fig/{type}/batch_{i}"):
                    os.makedirs(f"data_fig/{type}/batch_{i}")
                for j in range(0,128,16):
                    save_image(save_x[j],
                        os.path.join(f"data_fig/{type}/batch_{i}", 'origin_' + str(j) + '.png')
                    )
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch,
        train_loss * 128 / len(train_loader.dataset)))
    return


def test_epoch(model, test_loader, loss, epoch, type):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.reshape(128, 1, -1)
            z, x_ = model(x)
            test_loss += loss(x, x_).item()
            save_x = x.reshape(128, 28, 28)
            save_x_=x_.reshape(128,28,28)

            # 保存一些重构出来的图像用于(写报告)进一步研究 (5/100)
            with torch.no_grad():
                if i%30==0:
                    # 保存原始数据部分
                    if not os.path.exists(f"results/{type}/epoch_{epoch}/batch_{i}"):
                        os.makedirs(f"results/{type}/epoch_{epoch}/batch_{i}")
                    for j in range(0, 128, 16):
                        save_image(save_x_[j],
                            os.path.join(f"results/{type}/epoch_{epoch}/batch_{i}", 'train_' + str(j) + '.png')
                        )
                    # 保存训练结果部分
                    if not os.path.exists(f"datas/{type}/batch_{i}"):
                        os.makedirs(f"datas/{type}/batch_{i}")
                    for j in range(0, 128, 16):
                        save_image(save_x[j],
                            os.path.join(f"datas/{type}/batch_{i}", 'train_' + str(j) + '.png')
                        )
    test_loss*=128
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# def train(model, train_loader,test_loader, loss, optimizer, epoch_num, type):  #
#
#         # 保存一些重构出来的图像用于(写报告)进一步研究 (5/100)
#         # 保存图像的部分，参见train_epoch和test_epoch
#     return

def main(args):
    # encoder_args = {
    #     "x_dim": args.x_dim,
    #     "hidden_size": args.hidden_size,
    #     "latent_size": args.latent_size,
    #     "is_dist": True if args.type in ["VAE", "CVAE"] else False,
    # }
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
    # print("模型参数：", list((auto_encoder.parameters())))
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
                    optimizer=optimizer, epoch=epoch, type=args.type)
        test_epoch(model=auto_encoder,test_loader=test_loader, loss=loss, epoch=epoch, type=args.type)


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="AE", choices=["AE", "VAE", "CVAE"])
    parser.add_argument("--x_dim", default=784, type=int)  # 28 x 28 的像素展开为一个一维的行向量，每行代表一个图片
    parser.add_argument("--latent_size", default=10, type=int)  # 输出层大小，即服从高斯分布的隐含变量的维度。
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epoch_num", default=10, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
