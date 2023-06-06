"""
使用感知机（Perceptron）实现双月数据集的分类
"""

# -*- coding: UTF-8 -*- #
"""
@filename:double_moon.py
@author:201300086
@time:2023-03-21
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import Perceptron

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def moon(N, w, r, d):
    '''
    :param N: 半月散点数量
    :param w: 半月宽度
    :param r: x 轴偏移量
    :param d: y 轴偏移量
    :return: data (2*N*3) 月亮数据集 data_dn(2*N*1) 标签
    '''
    data = np.ones((2 * N, 4))
    np.random.seed(1919810)

    # 半月 1 的初始化
    r1 = 10  # 半月 1 的半径,圆心
    w1 = np.random.uniform(-w / 2, w / 2, size=N)  # 半月 1 的宽度范围
    theta1 = np.random.uniform(0, np.pi, size=N)  # 半月 1 的角度范围
    x1 = (r1 + w1) * np.cos(theta1)  # 行向量
    y1 = (r1 + w1) * np.sin(theta1)
    label1 = [1 for i in range(1, N + 1)]  # label for Class 1

    # 半月 2 的初始化
    r2 = 10  # 半月 2 的半径,圆心
    w2 = np.random.uniform(-w / 2, w / 2, size=N)  # 半月 2 的宽度范围
    theta2 = np.random.uniform(np.pi, 2 * np.pi, size=N)  # 半月 2 的角度范围
    x2 = (r2 + w2) * np.cos(theta2) + r
    y2 = (r2 + w2) * np.sin(theta2) - d
    label2 = [-1 for i in range(1, N + 1)]  # label for Class 2

    data[:, 1] = np.concatenate([x1, x2])
    data[:, 2] = np.concatenate([y1, y2])
    data[:, 3] = np.concatenate([label1, label2])
    return data


def plot_scatters(data):
    colors = ['darkred', 'green']
    lable = [1, -1]
    y = data[:, -1]
    assert (len(colors) >= len(lable))
    for i in range(len(lable)):
        plt.scatter(data[y == lable[i], 1],  # 横坐标
                    data[y == lable[i], 2],  # 纵坐标
                    c=colors[i],label=lable[i],s=1)


def Perceptron_train(lr, N, w, r, d):
    data = moon(N, w, r, d)
    perceptron = Perceptron(tol=1e-6, max_iter=1000, shuffle=True, eta0=lr,verbose=2)
    perceptron.fit(data[:, 1:3], data[:, -1])

    w = perceptron.coef_[0]  # 二维数组
    b = perceptron.intercept_
    acc = perceptron.score(data[:, 1:3], data[:, -1])
    print("准确率：",acc,"  迭代轮数：",perceptron.n_iter_)
    x_ticks = np.linspace(-10 - abs(r), 10 + abs(r), 100)
    plot_scatters(data)
    plt.plot(x_ticks, ((w[0] ) * x_ticks - b)/ w[1])
    plt.title("学习率：{:0>2f}".format(lr))


Perceptron_train(1e-4, 2000, 6, 10, -2)
plt.legend()
plt.show()
