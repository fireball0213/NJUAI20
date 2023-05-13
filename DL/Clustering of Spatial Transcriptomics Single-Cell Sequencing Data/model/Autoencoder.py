# -*- coding: UTF-8 -*- #
"""
@filename:Autoencoder
@author:201300086
@time:2023-02-04
"""
import numpy as np

# -*- coding: UTF-8 -*- #
"""
@filename:AE
@author:201300086
@time:2023-02-03
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model.dataset import load_h5


class autoencoder_Conv(nn.Module):
    def __init__(self):
        super(autoencoder_Conv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)#[16, 21, 193]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)#[8, 63, 579]
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)#[1, 124, 1156]
            nn.Tanh()
        )

    def forward(self, x):  # torch.Size([1, 128, 1159])
        # print(x.shape)
        encode = self.encoder(x)  # torch.Size([8, 10, 96])
        # print(encode.shape)
        decode = self.decoder(encode)  # torch.Size([1, 124, 1156])
        # print(decode.shape)
        # print()
        return encode, decode


class autoencoder(nn.Module):
    def __init__(self, init_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(init_size, 800),
            nn.Tanh(),
            # nn.Linear(800, 500),
            # nn.Tanh(),
            # nn.Linear(500, 200),
            # nn.Tanh(),
            # nn.Linear(200, 100),
            # nn.MaxPool2d(3, stride=1, padding=(1))
        )

        self.decoder = nn.Sequential(
            # nn.Linear(100, 200),
            # nn.Tanh(),
            # nn.Linear(200, 500),
            # nn.Tanh(),
            # nn.Linear(500,800),
            # nn.ReLU(True),
            nn.Linear(800, init_size),
            nn.ReLU(True),
        )

    def forward(self, x):  # torch.Size([1, 128, 1159])
        # print(x.shape)
        encode = self.encoder(x)  # torch.Size([8, 10, 96])
        # print(encode.shape)
        decode = self.decoder(encode)  # torch.Size([1, 124, 1156])
        # print(decode.shape)
        # print()
        return encode, decode


def AE_train(X, model_path, max_epoch):
    # 超参数设置
    batch_size = 128
    lr = 1e-2
    weight_decay = 1e-5
    epoches = max_epoch
    model = autoencoder(init_size=X.shape[-1])
    X = X.astype("float32")

    train_X = DataLoader(X, shuffle=False, batch_size=batch_size, drop_last=True)
    criterion = nn.MSELoss()
    # criterion =nn.L1Loss()

    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epoches):
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1
        for i, img in enumerate(train_X):
            img = img.reshape(1, batch_size, -1)

            # forward
            _, output = model(img)
            loss = criterion(output, img)

            # backward
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        print("epoch=", epoch, loss.data.float())
    # 保存模型
    torch.save(model.state_dict(), model_path)


def AE_reduction(X, model_path):
    load_size = 2
    model = autoencoder(init_size=X.shape[-1])
    X = X.astype("float32")
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    train_X = DataLoader(X, shuffle=False, batch_size=load_size, drop_last=True)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    X_reduction = []
    X_construction = []
    for i, img in enumerate(train_X):
        img = img.reshape(1, load_size, -1)
        encode_img = model.encoder(img)
        _, output = model(img)
        loss = criterion(output, img)
        # print(i, loss)

        X_reduction.append(encode_img[0][0].detach().numpy())
        X_reduction.append(encode_img[0][1].detach().numpy())
        X_construction.append(output[0][0].detach().numpy())
        X_construction.append(output[0][1].detach().numpy())
    X_reduction = np.array(X_reduction)
    X_construction = np.array(X_construction)
    print("reducton success ", len(X_reduction))
    return X_reduction, X_construction


if __name__ == "__main__":
    file_path = "spca_dat/sample_151507.h5"
    X, y, pos, lable = load_h5(file_path)
    AE_train(X, model_path="model/Linear_L2_10_02.pt")
    AE_reduction(X, model_path="model/Linear_L2_10_02.pt")
