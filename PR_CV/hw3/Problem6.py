# -*- coding: UTF-8 -*- #
"""
@filename:Problem6.py
@author:201300086
@time:2023-05-10
"""
# 6.f
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以便结果可复现
np.random.seed(0)
torch.manual_seed(0)

# Generate data
# 将0.25N(x; 0, 1) + 0.75N(x; 6, 4)作为原始数据分布,生成这个分布，从中抽样 10000 个样本点作为训练集

# train_data = np.random.normal(0, 1, 10000) * 0.25 + np.random.normal(6, 4, 10000) * 0.75
# valid_data = np.random.normal(0, 1, 1000) * 0.25 + np.random.normal(6, 4, 1000) * 0.75
# test_data = np.random.normal(0, 1, 1000) * 0.25 + np.random.normal(6, 4, 1000) * 0.75

# Number of samples
n_samples = 10000
n_samples_val_test = 1000
# Proportions
prop1 = 0.25
prop2 = 0.75

# Normal distributions
dist1 = np.random.normal(0, 1, int(n_samples * prop1))
dist2 = np.random.normal(6, 4, int(n_samples * prop2))
# For validation set
dist1_val = np.random.normal(0, 1, int(n_samples_val_test * prop1))
dist2_val = np.random.normal(6, 4, int(n_samples_val_test * prop2))


# For test set
dist1_test = np.random.normal(0, 1, int(n_samples_val_test * prop1))
dist2_test = np.random.normal(6, 4, int(n_samples_val_test * prop2))

# Combine data
train_data = np.concatenate((dist1, dist2))
valid_data = np.concatenate((dist1_val, dist2_val))
test_data = np.concatenate((dist1_test, dist2_test))
# print(train_data.shape)
# 转化为 PyTorch tensor
train_data = torch.Tensor(train_data)
val_data = torch.Tensor(valid_data)
test_data = torch.Tensor(test_data)
# 使用 DataLoader
train_loader = DataLoader(train_data, batch_size=50)
val_loader = DataLoader(valid_data, batch_size=50)
test_loader = DataLoader(test_data, batch_size=50)

# 查看dataloader中的数据
# for data in train_loader:
#     print(len(data))

# 定义自编码器
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x,flag=0):
        encoded = self.encoder(x)
        if flag=="train":
            # 编码结果添加噪声
            noise = torch.rand(encoded.shape) - 0.5
            encoded = encoded + noise
        else:
            # 测试阶段取整
            encoded = encoded.round()
        decoded = self.decoder(encoded)
        return decoded, encoded


# 定义训练函数

def train(model, train_loader, val_loader, epochs, l1_factor, bitrate_factor):
    model.train()
    min_val_loss = float('inf')  # 初始化最小验证损失为无穷大
    patience = 20  # 定义容忍次数
    no_improve_epochs = 0  # 记录验证损失没有改善的轮数

    for epoch in range(epochs):
        total_bits = 0
        total_samples = 0
        for data in train_loader:
            data = data.view(-1, 1)
            optimizer.zero_grad()
            output, encoded_data = model(data,flag="train")  # 前向传播时返回编码的数据

            # 计算 MSE 损失
            mse_loss = nn.MSELoss()(output, data)

            # 计算 L1 正则化损失
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))

            # 计算码率
            total_bits += torch.sum(encoded_data.view(-1).int()).item()
            total_samples += encoded_data.size(0)
            bitrate = total_bits / total_samples

            # 将 MSE 损失、L1 正则化损失和码率相加
            loss = mse_loss + l1_factor * l1_loss + bitrate_factor * bitrate

            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch + 1}, Train Loss: {loss.item()}, Bitrate: {bitrate}',end=',   ')
        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_data in val_loader:
                val_data = val_data.view(-1, 1).float()#
                val_output, _ = model(val_data)
                val_loss = nn.MSELoss()(val_output, val_data)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Validation MSE Loss: {avg_val_loss}')

        # 检查验证损失是否有改善
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # 如果验证损失连续patience轮没有改善，则提前停止训练
        if no_improve_epochs >= patience:
            print('Early stopping!')
            break

        # 回到训练模式
        model.train()


if __name__ == '__main__':
    encode_dim=1
    # 初始化模型和优化器
    model = Autoencoder(encode_dim)
    optimizer = optim.Adam(model.parameters())
    # 训练模型
    # train(model, train_loader, val_loader, epochs=300, l1_factor=0.01, bitrate_factor=0.1)
    # #保存模型
    # torch.save(model.state_dict(), 'autoencoder1.pth')

    # 加载模型
    model = Autoencoder(encode_dim)
    model.load_state_dict(torch.load('autoencoder1.pth'))

    # 在测试数据上评估模型
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.view(-1, 1)
            output, _ = model(data)
            test_loss += nn.MSELoss()(output, data).item()
    test_loss = test_loss / len(test_loader)
    print('Test Loss:', test_loss)

    # 对测试数据进行编码和解码
    with torch.no_grad():
        encoded_data = model.encoder(test_data.view(-1, 1)).numpy()
        encoded_data = encoded_data.round()
        decoded_data = model.decoder(torch.from_numpy(encoded_data)).numpy()


    # 输出原始数据，编码解码后的数据
    print("Original Data:", test_data[:5].numpy())
    print("Decoded Data:", decoded_data[:5].flatten())
    print("Encoded Data:", encoded_data[:5])


    # 画图
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    # 画编码的高斯分布直方图
    plt.hist(encoded_data, bins=50, label='Encoded Data', alpha=0.5,rwidth=0.5)

    #画test_data和decoded_data的全局分布直方
    # plt.hist(test_data.numpy(),bins=50, label='Original Data', alpha=0.5,rwidth=2)
    # plt.hist(decoded_data,bins=50, label='Decoded Data', alpha=0.5,rwidth=2)
    plt.xlabel('Data')
    plt.ylabel('Number of Samples')

    #使图中每个点更小
    # plt.rcParams['lines.markersize'] = 0.1
    # plt.plot(test_data.numpy()[:100], label='Original Data', c='r')
    # plt.plot(decoded_data[:100], label='Decoded Data')
    # plt.plot(encoded_data[:100], label='Encoded Data')

    plt.legend()
    plt.show()
