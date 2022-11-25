# -*- coding: UTF-8 -*- #
"""
@filename:my_operator.py
@author:201300086
@time:2022-11-23
"""
import networkx as nx
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import random

matplotlib.use('TkAgg')


def get_fitness(graph, x, n_edges, threshold=0):
    x_eval = np.where(x > threshold, 1, -1)
    # 获得Cuts值需要将图分为两部分, 这里默认以0为阈值把解分成两块.
    g1 = np.where(x_eval == -1)[0]
    g2 = np.where(x_eval == 1)[0]
    fitness = nx.cut_size(graph, g1, g2) / n_edges
    fitness = round(fitness, 6)
    return fitness  # 调用接口得到cut_size


def get_best_fitness(group):
    g = sorted(group, key=takeSecond)  # 按fitness升序
    return g[-1]


def one_bit_mutation(x):
    bit = random.randint(0, len(x) - 1)
    x[bit] = x[bit] * -1
    return x


def bit_wise_mutation(x, p):
    for num, i in enumerate(x):
        seed = random.randint(1, len(x))
        if seed < int(len(x) * p):
            x[num] *= -1
    return x


def one_point_crossover(x, y, p):
    seed = random.randint(0, len(x) - 1)
    if seed < int(len(x) * p):
        x1 = np.hstack((x[:seed], y[seed:]))
        y1 = np.hstack((y[:seed], x[seed:]))
        return x1, y1
    return x, y


def takeSecond(elem):
    return elem[1]


def fitness_propotional_selection(group, lamda, gama=0.5):
    """
    :param group: parent
    :param lamda: 用于交配父代个数，也是即将产生子代个数
    :return: 选出的parent集合
    """
    g = sorted(group, key=takeSecond)  # 按fitness升序
    pro = np.array(list(map(lambda x: x[1], g)))
    pro = pro - pro[0] * (gama - 0.001)
    pro = pro / pro.sum()
    new_group = []
    for i in range(lamda):
        index = np.random.choice(np.arange(len(g)), p=pro)
        new_group.append(g[index])
    return new_group


def survival_best_miu(newgroup, miu):
    g = sorted(newgroup, key=takeSecond)  # 按fitness升序
    return g[-miu:]
