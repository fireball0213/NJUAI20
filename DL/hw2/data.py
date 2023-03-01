import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision.datasets.mnist as mnist
import torchvision.transforms as transforms





import numpy as np
import math
import scipy.linalg as linalg
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def get_data(train=True, batch_size=128):
    dataset = mnist.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                #transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.1307,], [0.3081,],),
            ]
        ),
    )
    for i, (x, y) in enumerate(dataset):
        print(i,x.shape,y)
        #print(y.shape)
    return td.DataLoader(dataset, batch_size=batch_size,drop_last=True,)


if __name__ == "__main__":
    data=get_data()

    # for i, (x, y) in enumerate(data):
    #     print(i,x.shape)
    #     print(y.shape)
