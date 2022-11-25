# -*- coding: UTF-8 -*- #
"""
@filename:tools.py
@author:201300086
@time:2022-11-23
"""
import networkx as nx
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import random
import time

matplotlib.use('TkAgg')


def generate_binary(n):
    seed = "01"
    sa = []
    for i in range(n):
        sa.append(random.choice(seed))
    salt = ''.join(sa)
    return np.array(list(map(int, salt)))


def plot_performance(best_T, best_fit_list, label):
    plt.plot(best_T, best_fit_list, label=label)
    plt.xlabel('evaluations')
    plt.ylabel('fitness')
    # plt.title("Run on Graph regular")
    plt.legend(loc='best')
    plt.plot(best_T[-1], best_fit_list[-1], 'ks')
    show_max = str(best_fit_list[-1])
    plt.annotate(show_max, xytext=(best_T[-1] * 1.03, best_fit_list[-1]),
                 xy=(best_T[-1], best_fit_list[-1]),
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))

    # plt.show()


def plot_performance_realtime(best_T, best_fit_list, label):
    plt.clf()  # 清除之前画的图
    plt.plot(best_T, best_fit_list, label=label)
    plt.xlabel('evaluations')
    plt.ylabel('fitness')
    plt.title("Run on Graph regular")
    plt.legend(loc='best')
    plt.plot(best_T[-1], best_fit_list[-1], 'ks')
    show_max = str(best_fit_list[-1])
    plt.annotate(show_max, xytext=(best_T[-1] * 1.03, best_fit_list[-1]),
                 xy=(best_T[-1], best_fit_list[-1]),
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))
    plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
    plt.ioff()  # 关闭画图窗口
    # plt.show()


def view_fitness(group):
    lst = []
    for i in group:
        lst.append(i[1])
    print(lst)


def record_time(func):
    def wrapper(*args, **kwargs):  # 包装带参函数
        start_time = time.perf_counter()
        a = func(*args, **kwargs)  # 包装带参函数
        end_time = time.perf_counter()
        print('time=', end_time - start_time)
        return a  # 有返回值的函数必须给个返回

    return wrapper
