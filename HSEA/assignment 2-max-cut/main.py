# -*- coding: UTF-8 -*- #
"""
@filename:main.py
@author:201300086
@time:2022-11-23
"""
import networkx as nx
import numpy as np
import copy
import time
import argparse
import matplotlib
import matplotlib.pyplot as plt
import random

matplotlib.use('TkAgg')
from graph import graph_generator
from my_operator import bit_wise_mutation, one_bit_mutation, get_fitness, \
    one_point_crossover, fitness_propotional_selection, survival_best_miu, get_best_fitness
from tools import generate_binary, plot_performance, view_fitness, record_time, plot_performance_realtime


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-type', type=str, help='graph type', default='gset')
    parser.add_argument('--n-nodes', type=int, help='the number of nodes', default=5000)
    parser.add_argument('--n-d', type=int, help='the number of degrees for each node', default=5)
    parser.add_argument('--T', type=int, help='the number of fitness evaluations', default=10000)
    parser.add_argument('--seed-g', type=int, help='the seed of generating regular graph', default=1)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gset-id', type=int, default=1)
    parser.add_argument('--sigma', type=float, help='hyper-parameter of mutation operator', default=.1)
    parser.add_argument('--miu', type=int, help='the size of the group', default=4)
    parser.add_argument('--lamda', type=int, help='the number per parent selection', default=2)
    parser.add_argument('--p-m', type=int, help='the propobility of bit-wise mutation', default=0.05)
    parser.add_argument('--p-c', type=int, help='the propobility of one-point crossover', default=1)
    parser.add_argument('--gama', type=int, help='for FPS', default=0.5)
    args = parser.parse_known_args()[0]  # parser.parse_args()
    return args


def real_number(args=get_args()):
    # print(args)
    graph, n_nodes, n_edges = graph_generator(args)
    # print(graph)
    np.random.seed(args.seed)
    x = np.random.rand(n_nodes)  # 解，这里x使用实数值表示, 也可以直接使用01串表示, 并使用01串上的交叉变异算子
    best_fitness = get_fitness(graph, x, n_edges)  # 除上n_edges让不同规模图得到均等fitness
    best_T = []
    best_fit_list = []
    for i in range(args.T):  # 实数表示所以算法用简单的(1+1)ES
        tmp = x + np.random.randn(n_nodes) * args.sigma  # 每一轮用sigma*正态分布做变异算子
        tmp_fitness = get_fitness(graph, tmp, n_edges)
        if tmp_fitness > best_fitness:
            x, best_fitness = tmp, tmp_fitness
            print(i, best_fitness)
        best_T.append(i)
        best_fit_list.append(best_fitness)
    plot_performance(best_T, best_fit_list, label="baseline:real-number")


def binary_string_individual(args=get_args()):
    graph, n_nodes, n_edges = graph_generator(args)
    x = generate_binary(n_nodes)
    best_fitness = get_fitness(graph, x, n_edges)
    x = np.where(x > 0, 1, -1)
    best_T = [0]
    best_fit_list = [best_fitness]

    for i in range(args.T):  # 01串表示
        # tmp = one_bit_mutation(x)
        tmp = bit_wise_mutation(x, args.p_m)
        tmp_fitness = get_fitness(graph, tmp, n_edges)
        if tmp_fitness > best_fitness:
            x, best_fitness = tmp, tmp_fitness
            print("miu lamda p_m p_c γ:", i, best_fitness, args.miu, args.lamda, args.p_m, args.p_c, args.gama)
        best_T.append(i)
        best_fit_list.append(best_fitness)
    plot_performance(best_T, best_fit_list, label="bit_wise_mutation")


@record_time
def binary_string_group(args=get_args(), p_m=0.0035, p_c=0.6, miu=5, lamda=2, gama=1.0):
    graph, n_nodes, n_edges = graph_generator(args)
    group = []  # 记录当前种群中个体

    # 种群初始化
    for i in range(miu):
        x = generate_binary(n_nodes)
        init_best_fitness = get_fitness(graph, x, n_edges)
        x = np.where(x > 0, 1, -1)
        group.append((x, init_best_fitness))
    # print(group[0][1])

    # 迭代
    best_T = [0]
    best_fitness = get_best_fitness(group)[1]
    best_fit_list = [best_fitness]
    start_time = time.perf_counter()
    improve_k = 0

    # plt.title("Run on Graph regular")
    # plt.ion()
    for k in range(args.T):
        improve_k += 1
        # 设置时间上限
        # now_time=time.perf_counter()
        # if(now_time-start_time>200):
        #     print("evaluations: ",k)
        #     break
        old_group = copy.deepcopy(group)
        new_group = []
        select_group = fitness_propotional_selection(group, lamda, gama)
        for i in range(len(select_group)):
            tem1_x, tem2_x = one_point_crossover(select_group[i][0],
                                                 select_group[(i + 1) % len(select_group)][0], p_c)
            tem1_x = bit_wise_mutation(tem1_x, p_m)
            tem2_x = bit_wise_mutation(tem2_x, p_m)
            new_group.append((tem1_x, get_fitness(graph, tem1_x, n_edges)))  # 添加crossover后两个子代解到新种群
            new_group.append((tem2_x, get_fitness(graph, tem2_x, n_edges)))

        # 生存选择:fitness_based
        new_group.extend(old_group)
        group = survival_best_miu(new_group, miu)

        # 更新最优
        tmp_best_fitness = get_best_fitness(group)[1]
        if tmp_best_fitness > best_fitness:
            best_fitness = tmp_best_fitness
            print("SGA miu lamda p_m p_c γ d_k:", k, best_fitness, miu, lamda, p_m, p_c, gama, improve_k)
            improve_k = 0
            best_T.append(k)
            best_fit_list.append(best_fitness)
        elif improve_k > 10000:
            break
        # plot_performance_realtime(best_T, best_fit_list, label="SGA,p_m={},p_c={},μ={},λ={},γ={}".format(p_m, p_c, miu, lamda, gama))
    plot_performance(best_T, best_fit_list, label="SGA,p_m={},p_c={},μ={},λ={},γ={}".format(p_m, p_c, miu, lamda, gama))
    return best_fitness


def find_best_pro(args=get_args()):
    # p = []
    # fi = []
    for pro in range(1, 11, 3):
        pro /= 10
        pro = round(pro, 1)
        f = binary_string_group(args, gama=pro)
        # p.append(pro)
        # fi.append(f)
        print(pro, f)

    plt.show()


def find_best_miu_lamda(args=get_args()):
    m = []
    l = []
    fi = []
    for miu in range(5, 6, 1):
        # for lamda in range(miu-2,miu+1,1):
        lamda = int(miu - 3)
        f = binary_string_group(args, miu=miu, lamda=lamda)
        m.append(miu)
        fi.append(f)
        print(miu, lamda, f)
    plt.show()


def main(args=get_args()):
    # plt.subplot(211)
    plt.title("Run on Graph {}".format(args.gset_id))
    # plt.title("Run on Graph regular")
    # real_number(args)
    # plt.subplot(212)

    # binary_string_individual(args)
    # binary_string_group(args, miu=5, lamda=2, p_m = 0.025, p_c = 0.83, gama = 0.5)
    binary_string_group(args, miu=5, lamda=2, )
    binary_string_group(args, miu=5, lamda=2, )
    binary_string_group(args, miu=5, lamda=2, )
    # binary_string_group(args, miu=5, lamda=4,)
    # binary_string_group(args, miu=5, lamda=4, )

    # find_best_pro(args)
    # find_best_miu_lamda(args)
    plt.show()


if __name__ == '__main__':
    main()
