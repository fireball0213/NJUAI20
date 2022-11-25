# -*- coding = utf-8 -*-
# Time : 2022/5/28 17:19
# Author : 201300086史浩男
# File : Vectorization.py
# Software : PyCharm
import numpy as np
import time

def plain_distance_function(X):
    # 直观的距离计算实现方法
    # 首先初始化一个空的距离矩阵D
    D = np.zeros((X.shape[0], X.shape[0]))
    # 循环遍历每一个样本对
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            # 计算样本i和样本j的距离
            D[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
    return D

def plain_permutation_function(X, p):
    # 初始化结果矩阵, 其中每一行对应一个样本
    permuted_X = np.zeros_like(X)
    for i in range(X.shape[0]):
        # 采用循环的方式对每一个样本进行重排列
        permuted_X[i] = X[p[i]]
    return permuted_X

def matrix_distance_function(X: np.ndarray):
    # (xi - xj)^2 = xi^2 + xj^2 - 2 xi * xj
    ones = np.ones(X.shape).T
    M_i = np.square(X) @ ones#从m*d得到m*m的行平方和，每行都相同
    M_j = M_i.T
    M_ij = X @ X.T
    M = M_i + M_j - 2 * M_ij + 1e-10# 加1e-10防止对负数开根号
    return np.sqrt(M)


def matrix_permutation_function(X: np.ndarray, p: np.ndarray):
    M_per = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        M_per[i, p[i]] = 1
    return M_per @ X


def testspeed_distance_function(m: int, d: int,n:int,distance_function):
    sumtime=0
    for i in range(n):
        X = np.random.rand(m, d)#m行n列随机矩阵
        start_time = time.time()
        distance_function(X)
        end_time = time.time()
        sumtime+=end_time - start_time
    return sumtime


def testspeed_permutation_function(m: int, d: int,n:int, permutation_function):
    sumtime=0
    for i in range(n):
        X = np.random.rand(m, d)
        p = np.random.permutation(m)
        start_time = time.time()
        permutation_function(X, p)
        end_time = time.time()
        sumtime+=end_time - start_time
    return sumtime


def main():
    time_small_plain = testspeed_distance_function(m=10, d=10,n=100, distance_function=plain_distance_function)
    time_small_matrix = testspeed_distance_function(m=10, d=10,n=100, distance_function=matrix_distance_function)
    time_large_plain = testspeed_distance_function(m=1000, d=1000,n=1, distance_function=plain_distance_function)
    time_large_matrix = testspeed_distance_function(m=1000, d=1000,n=1 ,distance_function=matrix_distance_function)
    print('100 times for (10,10) scale using plain function:', time_small_plain)
    print('100 times for (10,10) scale using matrix function:', time_small_matrix)
    print('1 times for (1000,1000) scale using plain function:', time_large_plain)
    print('1 times for (1000,1000) scale using matrix function:', time_large_matrix)
    print("over")
    time_for_small_and_plain = testspeed_permutation_function(m=10, d=10,n=100,permutation_function=plain_permutation_function)
    time_for_small_and_matrix = testspeed_permutation_function(m=10, d=10,n=100,permutation_function=matrix_permutation_function)
    time_for_large_and_plain = testspeed_permutation_function(m=2000, d=2000,n=10,permutation_function=plain_permutation_function)
    time_for_large_and_matrix = testspeed_permutation_function(m=2000, d=2000, n=10,permutation_function=matrix_permutation_function)
    print('100 times for (10,10) scale using plain function:', time_for_small_and_plain)
    print('100 times for (10,10) scale using matrix function:', time_for_small_and_matrix)
    print('10 times for (2000,2000) scale using plain function:', time_for_large_and_plain)
    print('10 times for (2000,2000) scale using matrix function:', time_for_large_and_matrix)
    print("over")
main()