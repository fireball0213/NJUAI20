# -*- coding: UTF-8 -*- #
"""
@filename:k_means.py
@author:201300086
@time:2023-02-01
"""
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plot import plot_scatters
from model.dataset import load_h5, pooling_data
from utils import ARI, NMI, rotate_angle

matplotlib.use('TkAgg')
from Autoencoder import AE_reduction, AE_train


# 计算欧拉距离,返回每个点到质点的距离len(dateSet)*k的数组
def L2(dataSet, centroids, k, rotate=None):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids  # 沿y方向复制k倍

        if rotate != None:  # 将前两维实际坐标旋转，用于削弱非平行线方向的距离影响
            diff = rotate_angle(rotate, diff)
            diff[::, 1] = diff[::, 1] * 0  # 丢掉第二列y坐标
        clalist.append(np.sum(diff ** 2, axis=1) ** 0.5)  # axis=1表示行
    clalist = np.array(clalist)
    return clalist


def L1(dataSet, centroids, k, rotate=None):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids  # 沿y方向复制k倍
        if rotate != None:  # 将前两维实际坐标旋转，用于削弱非平行线方向的距离影响
            diff = rotate_angle(rotate, diff)
            diff[::, 1] = diff[::, 1] * 0  # 丢掉第二列y坐标

        clalist.append(np.sum(np.abs(diff), axis=1))  # axis=1表示行
    clalist = np.array(clalist)
    return clalist


# 计算质心
def classify(dataSet, centroids, k, distance_func=L2, rotate=None):
    clalist = distance_func(dataSet, centroids, k, rotate)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean()
    # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids


# 使用k-means分类
def kmeans(dataSet, k, max_eva, distance_func=L2, rotate=None):
    # 随机取质心
    centroids = random.sample(dataSet, k)
    evaluations = 0
    if rotate == None:
        print("no rotate")
    else:
        print("rotate:", rotate)
    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k, distance_func, rotate)
    print("kmeans_evaluations:", end="")
    while np.any(changed != 0) and evaluations <= max_eva:
        evaluations += 1
        changed, newCentroids = classify(dataSet, newCentroids, k, distance_func, rotate)
        print(evaluations, end=" ")
    print()
    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    # cluster = [[] for i in range(k)]
    cluster = []
    lable = []
    clalist = distance_func(dataSet, centroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)  # 遍历所有点，计算距离哪个质心最近
    for i, j in enumerate(minDistIndices):
        # cluster[j].append(dataSet[i])
        cluster.append(dataSet[i])
        lable.append(j)
    cluster = np.array(cluster)
    lable = np.array(lable)
    return centroids, cluster, lable


def count_same(angle, clu, label, x):  # 旋转angle后的clu在x=x上有多少个点
    cluster = rotate_angle(angle, clu)
    dic = {}
    for i in range(len(cluster)):
        if (cluster[i][0] >= x - 1 and cluster[i][0] <= x + 1):
            if label[i] in dic.keys():
                dic[label[i]] += 1
            else:
                dic[label[i]] = 1
    # print(angle,dic)
    return np.array(list(dic.values())).max()


def AE_kmeans(file_path, model_path, max_epoch, AE, train_model):
    X, y, pos, lable = load_h5(file_path)
    # dataset=list(pos)

    # 不用AE
    if AE == False:
        X = pooling_data(X, pos, 6, 2)
        dataset = list(np.hstack((pos, X)))  # 拼接
    else:
        X = pooling_data(X, pos, 10, 20000)
        # 训练模型
        if train_model == True:
            AE_train(X, model_path=model_path, max_epoch=max_epoch)
        # 读取模型
        X_reduction, X_construction = AE_reduction(X, model_path=model_path)

        dataset = list(np.hstack((pos, X_construction)))

    # 得到转角后再算一遍
    best_lie = 0.38
    centroids, cluster, label = kmeans(dataset, 8, max_eva=100, distance_func=L2, rotate=best_lie)
    # centroids, cluster, label = kmeans(dataset, 8, max_eva=100, distance_func=L2)

    # print('质心为：%s' % centroids)
    ARI_loss = ARI(y, label)
    NMI_loss = NMI(y, label)
    print("损失为:ARI=", ARI_loss, "  NMI=", NMI_loss)

    plt.axes([0.07, 0.1, 0.4, 0.8])
    plot_scatters(label, cluster, np.array(list(set(label))))
    for j in range(len(centroids)):
        plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50)
    plt.axes([0.57, 0.1, 0.4, 0.8])
    plot_scatters(y, pos, lable)
    plt.show()


if __name__ == '__main__':
    file_path = "spca_dat/sample_151507.h5"
    model_path = "save_model/Linear_L2_100_06.pt"
    AE_kmeans(file_path, model_path, max_epoch=100, AE=True, train_model=False)

    # 预聚类，计算平行线转角
    # centroids, cluster,label = kmeans(dataset, 8,max_eva=5,distance_func=L2)
    # ARI_loss=ARI(y,label)
    # NMI_loss=NMI(y,label)
    # print("损失为:ARI=",ARI_loss,"  NMI=",NMI_loss)
    # lie = np.linspace(0.32,0.4, 20)
    # best_lie=0
    # best_count=0
    # for i in lie:
    #     #new_cluster=rotate_angle(i,cluster)
    #     count=count_same(i,cluster,label,0)
    #     if count>best_count:
    #         best_lie=i
    #         best_count=count
    # #cluster = rotate_angle(best_lie, cluster)
    # print("best angle=",best_lie," max count=",best_count)
    # plt.axes([0.07, 0.5, 0.2, 0.4])
    # plot_scatters(label, cluster, np.array(list(set(label))))
