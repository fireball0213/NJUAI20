import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision.datasets.mnist as mnist
import torchvision.transforms as transforms


def get_data(train=True, batch_size=128):
    dataset = mnist.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.1307,], [0.3081,],),
            ]
        ),
    )
    return td.DataLoader(dataset, batch_size=batch_size)
