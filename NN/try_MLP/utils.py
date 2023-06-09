# -*- coding: UTF-8 -*- #
"""
@filename:utils.py
@author:201300086
@time:2023-06-06
"""
import torchvision
import torch.utils.data as Data
import numpy as np
import struct
import os
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import time
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
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


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def plot_mnist(path, flag='train'):
    train_images, train_labels = load_mnist(path, kind='train')
    test_images, test_labels = load_mnist(path, kind='t10k')

    # 查看数据集
    fig = plt.figure(figsize=(8, 8))
    plt.title(flag)
    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        if flag == 'train':
            images = np.reshape(train_images[i], [28, 28])
            ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
            ax.text(0, 7, str(train_labels[i]))
        else:
            images = np.reshape(test_images[i], [28, 28])
            ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
            ax.text(0, 7, str(test_labels[i]))
    plt.tight_layout()
    plt.show()

    # 查看标记的分布
    ax = fig.add_subplot(1, 1, 1)
    if flag == 'train':
        ax.hist(train_labels, bins=10)
    else:
        ax.hist(test_labels, bins=10)
    plt.show()


def get_one_hot(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def dataset_loader():
    path = 'MNIST/MNIST/raw'
    print('Load data from ', path, end=' ')
    train_image, train_label = load_mnist(path, kind='train')
    test_image, test_label = load_mnist(path, kind='t10k')

    # 必须归一化
    train_image = train_image.astype(np.float32) / 255.0
    # train_label = get_one_hot(train_label)
    # train_label = train_label.reshape(train_label.shape[0], train_label.shape[1], 1)

    test_image = test_image.astype(np.float32) / 255.0
    # test_label = get_one_hot(test_label)
    # test_label = test_label.reshape(test_label.shape[0], test_label.shape[1], 1)
    print('Load data success!')
    return train_image, train_label, test_image, test_label


if __name__ == "__main__":
    path = 'MNIST/MNIST/raw'
    plot_mnist(path, flag='train')
    plot_mnist(path, flag='test')
