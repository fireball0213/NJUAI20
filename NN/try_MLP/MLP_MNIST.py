"""
使用MNIST数据集，训练一个MLP模型，包含两个全连接层，第一层包含2048个神经元，第二层包含10个神经元，激活函数为ReLU。
SGD优化器，学习率0.001，5个epoch，batch size为100。
1.探究不同参数初始化方法（Default,Uniform,Normal,Xavier,Kaiming）对模型训练的影响。
2.探究不同的归一化方法（BN、LN、IN、GN)对模型训练的影响。
3.探究归一化方法和参数初始化方法有哪些复杂的相互作用
"""

# -*- coding: UTF-8 -*- #
"""
@filename:MLP_MNIST.py
@author:201300086
@time:2023-06-06
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import time
from utils import timer, evaluate, Plot_data,get_MNIST

class MLP(nn.Module):
    def __init__(self, norm):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(784, 2048)
        self.layer2 = nn.Linear(2048, 10)
        self.relu = nn.ReLU()
        if norm == 'BN':
            self.norm1 = nn.BatchNorm1d(2048)
            self.norm2 = nn.BatchNorm1d(10)
        elif norm == 'LN':
            self.norm1 = nn.LayerNorm(2048)
            self.norm2 = nn.LayerNorm(10)
        elif norm == 'IN':
            self.norm1 = nn.InstanceNorm1d(2048)
            self.norm2 = nn.InstanceNorm1d(10)
        elif norm == 'GN':
            self.norm1 = nn.GroupNorm(32, 2048)
            self.norm2 = nn.GroupNorm(2, 10)

    def forward(self, input, norm):
        out = self.layer1(input)
        if norm is not None:
            out = self.norm1(out)
        out = self.relu(out)
        out = self.layer2(out)
        if norm is not None:
            out = self.norm2(out)
        return out


def train(model, train_loader, val_loader, num_epochs, criterion, optimizer, norm, scheduler, flag):
    print(f"Train on {device}")
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        a1 = time.time()
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1), norm)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        val_accuracy = evaluate(model, val_loader, norm)
        a2 = time.time()
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%,"
            f" Val Acc: {val_accuracy:.2f}%, LR:{optimizer.param_groups[0]['lr']},"f" time cost:{a2 - a1}")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    # 保存一次训练中的所有关键信息，输出到Log文件夹中，文件包含本次运行时间戳
    with open('./Log/Log_' + str(time.time()) + '.txt', 'w') as f:
        f.write('Init: ' + str(flag) + '\n')
        f.write('Norm: ' + str(norm) + '\n')
        f.write('Train Loss:\n')
        for i in train_losses:
            f.write(str(i) + '\n')
        f.write('Train Accuracy:\n')
        for i in train_accuracies:
            f.write(str(i) + '\n')
        f.write('Val Accuracy:\n')
        for i in val_accuracies:
            f.write(str(i) + '\n')
    return train_losses, train_accuracies, val_accuracies


def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if m.bias is not None:
            m.bias.data.uniform_(-0.1, 0.1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.normal_(0.0, 0.02)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def weights_init_he(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


def my_model(flag, norm=None):
    train_loader, val_loader = get_MNIST()

    model = MLP(norm).to(device)
    # 5种初始化方法
    print("Initialization method: ", flag, " Norm: ", norm)
    if flag == "Default Initialization":
        pass
    elif flag == "Uniform Initialization":
        model.apply(weights_init_uniform)
    elif flag == "Normal Initialization":
        model.apply(weights_init_normal)
    elif flag == "Xavier Initialization":
        model.apply(weights_init_xavier)
    elif flag == "He Initialization":
        model.apply(weights_init_he)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train and plot the charts
    num_epochs = 5
    train_losses, train_accuracies, val_accuracies = train(model, train_loader, val_loader, num_epochs, criterion,
                                                           optimizer, scheduler=None, norm=norm, flag=flag)
    # Plot the charts
    Plot_data(range(1, num_epochs + 1), train_losses, train_accuracies, val_accuracies, label=flag, norm=norm)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 探究不同的初始化方法、不同归一化方法对模型的影响
    norm = 'GN'  # ['BN' 'LN' 'IN' 'GN' None]
    my_model("Default Initialization", norm=norm)
    my_model("Uniform Initialization", norm=norm)
    my_model("Normal Initialization", norm=norm)
    my_model("Xavier Initialization", norm=norm)
    my_model("He Initialization", norm=norm)
