# -*- coding: UTF-8 -*- #
"""
@filename:data.py
@author:201300086
@time:2023-01-29
"""
import numpy as np


def read_data_regression(path):
    data = np.loadtxt(path, delimiter=",", dtype=object)
    Z = data.T[-1]
    for i in range(len(Z)):
        if Z[i] == "g":
            Z[i] = 1.0
        else:
            Z[i] = 2.0
    data = data.T[:-1].T.astype("float32")  # 字符串转float
    # 计算初始系数矩阵Alpha
    DATA = data
    Alpha = np.linalg.pinv(DATA) @ Z
    Alpha = Alpha.astype("float32").T
    return Z, DATA, Alpha


def read_data_cover(file):
    # data = np.loadtxt(file, delimiter=" ", dtype=int)
    # data=data[::,:-1]
    # data=np.load("G500.npy")
    data = np.load("data/G_dic.npy", allow_pickle=True).item()
    return data


def cut_data(file, n):
    data = np.loadtxt(file, delimiter=" ", dtype=int)
    data = data[::, :-1]
    new_data = []
    for i in range(len(data)):
        if data[i][0] not in range(n + 1, 801) and data[i][1] not in range(n + 1, 801):
            new_data.append(data[i])
    new_data = np.array(new_data)
    print(new_data.shape)
    np.save("G500", new_data)


def dic_data(file):
    data = read_data_cover(file)
    dic = {}
    for i in range(1, 501):
        dic[i] = []
    for i in range(len(data)):
        dic[data[i][0]].append(data[i][1])
        dic[data[i][1]].append(data[i][0])
    np.save("G_dic", dic)
    return dic

# file="G500.npy"
# read_data_cover(file)
