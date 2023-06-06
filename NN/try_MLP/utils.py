# -*- coding: UTF-8 -*- #
"""
@filename:utils.py
@author:201300086
@time:2023-06-06
"""
import torchvision
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 一个记录函数运行时间的装饰器函数
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print('time cost:{}'.format(end_time - start_time))

    return wrapper

def evaluate(model, data_loader, norm):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs.view(inputs.size(0), -1), norm)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def Plot_data(epochs, train_losses, train_accuracies, val_accuracies, label: str, norm=None):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(label + ', Norm: ' + str(norm))
    plt.plot(epochs, train_losses, label='Train')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')

    plt.subplot(1, 2, 2)
    plt.title(label + ', Norm: ' + str(norm))
    plt.plot(epochs, train_accuracies, label='Train')
    plt.plot(epochs, val_accuracies, label='Valid')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()


def plot_boston_loss(train_losses, criterion, learning_rate):
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('loss_func=' + str(criterion) + ',lr=' + str(learning_rate))
    plt.legend()
    plt.show()


def plot_boston_pred(y_val_tensor, val_outputs, criterion, learning_rate):
    plt.plot(y_val_tensor[:, 0], label='True Values')
    plt.plot(val_outputs.flatten(), label='Predictions')
    plt.xlabel('True Values')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Values with ' + 'loss_func=' + str(criterion) + ',lr_init=' + str(learning_rate))
    plt.show()


def get_MNIST():
    train_data = torchvision.datasets.MNIST(
        root='MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_data = torchvision.datasets.MNIST(
        root='MNIST',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    batch_size = 100
    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader