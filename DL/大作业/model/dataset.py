# -*- coding: UTF-8 -*- #
"""
@filename:dataset.py
@author:201300086
@time:2023-02-01
"""
import copy

import deepdish as dd
import numpy as np

file_path = "spca_dat/sample_151508.h5"


def load_h5(file_path):
    data = dd.io.load(file_path)
    X = data["X"]
    Y = data["Y"]
    pos = data["pos"].T
    # print(data["X"],data["X"].shape)
    # print(data["Y"],data["Y"].shape)
    # print(pos,pos.shape)
    y = []
    y_dic = {}
    for i in Y:
        i = str(i)[-3:-1]
        y.append(i)
        if i in y_dic.keys():
            y_dic[i] += 1
        else:
            y_dic[i] = 1
    # print(y_dic)
    # print(Y)
    y = np.array(y)
    # print(y)
    lable = list(y_dic.keys())
    lable.sort()
    # print(lable)
    print("load data success!", file_path)
    # print(np.hstack((pos,X)),np.hstack((pos,X)).shape)
    return X, y, pos, lable


def pooling_data(X, pos, k, alpha):
    """
    :param X: 对X矩阵作k*k的平均池化
    :param pos:
    :param k: 把附近k*k个点的平均基因表达，乘上权重系数后，作为当前点的第三维坐标向量，然后用这个三维坐标的欧式距离作为聚类标准
    :param alpha: 池化矩阵扩大倍数，用于放大基因影响（减少位置影响）
    :return: 池化后的X矩阵
    """
    Y = copy.deepcopy(X)
    # 字典存储位置和X值的对应关系
    dic = {}
    for i in range(len(X)):
        dic[tuple(pos[i])] = X[i]

    for i in range(len(X)):  # 针对每个点计算一圈近邻
        x, y = pos[i][0], pos[i][1]
        count = 1  # 近邻个数,包括自己
        for p in range(x - k + 1, x + k):
            for q in range(y - k + 1, y + k):
                if tuple((p, q)) in dic.keys():  # 近邻真实存在
                    count += 1
                    Y[i] += dic[tuple((p, q))]
        Y[i] *= alpha
        Y[i] /= count
    return Y


if __name__ == "__main__":
    X, y, pos, lable = load_h5(file_path)
    # pooling_data(X,pos,2)
