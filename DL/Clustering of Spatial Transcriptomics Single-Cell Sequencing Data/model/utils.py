# -*- coding: UTF-8 -*- #
"""
@filename:utils.py
@author:201300086
@time:2023-02-01
"""
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import math
import time

def ARI(labels_true,labels_pred ):
    return adjusted_rand_score(labels_true, labels_pred)

def NMI(labels_true,labels_pred ):
    return normalized_mutual_info_score(labels_true,labels_pred, average_method='arithmetic')


def rotate_angle(angle, pos_list):  # 旋转xy坐标
    new_pos = pos_list.copy()
    valuex = pos_list.T[0].copy()
    valuey = pos_list.T[1].copy()
    new_pos.T[0] = math.cos(angle) * valuex - math.sin(angle) * valuey
    new_pos.T[1] = math.cos(angle) * valuey + math.sin(angle) * valuex
    return new_pos

def record_time(func):
    def wrapper(*args, **kwargs):  # 包装带参函数
        start_time = time.perf_counter()
        a = func(*args, **kwargs)  # 包装带参函数
        end_time = time.perf_counter()
        print('time=', end_time - start_time)
        return a  # 有返回值的函数必须给个返回

    return wrapper
if __name__ == "__main__":
    labels_true = [0, 0, 0, 1, 1, 1]
    labels_pred = [0, 0, 1, 1, 2, 2]
    print(ARI(labels_true, labels_pred))

    C = [1, 1, 2, 2, 3, 3, 3]
    D = [2,2,3,3, 1, 1, 1]
    print(NMI(C, D))