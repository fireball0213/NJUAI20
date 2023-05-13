# -*- coding: UTF-8 -*- #
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import time

matplotlib.use('TkAgg')
from sklearn.metrics import mean_squared_error
from tools.dataset import read_data_regression, read_data_cover


def record_time(func):
    def wrapper(*args, **kwargs):  # 包装带参函数
        start_time = time.perf_counter()
        a = func(*args, **kwargs)  # 包装带参函数
        end_time = time.perf_counter()
        print('time=', end_time - start_time)
        return a  # 有返回值的函数必须给个返回

    return wrapper


def generate_binary(n, problem="regression"):
    seed = "01"
    sa = []
    for i in range(n):
        sa.append(random.choice(seed))
    salt = ''.join(sa)
    x = np.array(list(map(int, salt)))
    if problem == "cover":
        now = sum(x)
        q = (now - k - 1) / now
        for num, i in enumerate(x):
            seed = random.random() * len(x)
            if seed < int(len(x) * q):
                if x[num] == 1:
                    x[num] = 0
    return x


k = 8


def clear_solution(solution_group, problem):
    group = []  # 去重后满足2k的
    group2 = []  # 去重的
    dic = {}
    if problem == "regression":
        for n in solution_group:
            if f1(n) in dic.keys():
                pass
            else:
                dic[f1(n)] = 1
                group2.append(n)
        for i in group2:
            if f2(i) <= 2 * k:
                group.append(i)
    elif problem == "cover":
        for n in solution_group:
            if f3(n) in dic.keys():
                pass
            else:
                dic[f3(n)] = 1
                group2.append(n)
        for i in group2:
            if f4(i) <= 2 * k:
                group.append(i)
    else:
        assert (1 < 0)

    if len(group) > 3:
        return group
    else:
        return group2


def one_bit_mutation(x):
    bit = random.randint(0, len(x) - 1)
    x[bit] = x[bit] * -1
    return x


def bit_wise_mutation(y, p):
    x = copy.deepcopy(y)  # 必须先copy再操作
    for num, i in enumerate(x):
        seed = random.randint(0, len(x))
        if seed < int(len(x) * p):
            if x[num] == 0:
                x[num] = 1
            else:
                x[num] = 0
    return x


# @record_time
def bit_wise_mutation_cover(y, p):
    x = copy.deepcopy(y)  # 必须先copy再操作
    for num, i in enumerate(x):
        seed = random.random() * len(x)
        if seed < int(len(x) * p):
            if x[num] == 0:
                x[num] = 1
            else:
                x[num] = 0
    # 控制解中1数量不要太多
    now = sum(x)
    q = (now - k - 1) / now
    for num, i in enumerate(x):
        seed = random.random() * len(x)
        if seed < len(x) * q:
            if x[num] == 1:
                x[num] = 0
    return x


def one_point_crossover(x, y, p):
    seed = random.randint(0, len(x) - 1)
    if seed < int(len(x) * p):
        x1 = np.hstack((x[:seed], y[seed:]))
        y1 = np.hstack((y[:seed], x[seed:]))
        return x1, y1
    return x, y


path = "data/ionosphere.data"
Z, DATA, Alpha = read_data_regression(path)


def f1(solution):  # 计算稀疏回归的MSE
    new_Alpha = []
    for i in range(len(solution)):
        if solution[i] == 0:
            new_Alpha.append(0)
        else:
            new_Alpha.append(Alpha[i])
    new_result = DATA @ new_Alpha
    return mean_squared_error(Z, new_result)


def f2(solution):  # 计算稀疏回归的S元素个数
    return sum(solution)


# file="G1.txt"
# file="G500.npy"
file = "data/G_dic.npy"
data = read_data_cover(file)


# @record_time
def f3(solution):  # 计算最大覆盖点集合,max
    # 查找有指定编号的行，添加到集合
    cover_set = []
    for i in range(1, 501):
        if solution[i - 1] == 0:
            pass
        else:
            cover_set.extend(data[i])
    cover_set = set(cover_set)
    return len(cover_set)


def f4(solution):  # 点个数,min
    return sum(solution)



def plot_two_f(f1_values, f2_values, front):
    # front.reverse()
    front0_1 = []
    front1_1 = []
    front2_1 = []
    front0_2 = []
    front1_2 = []
    front2_2 = []
    for index in front[0]:
        front0_1.append(f1_values[index])
        front0_2.append(f2_values[index])
    for index in front[1]:
        front1_1.append(f1_values[index])
        front1_2.append(f2_values[index])
    for index in front[2]:
        front2_1.append(f1_values[index])
        front2_2.append(f2_values[index])
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(front0_1, front0_2, label='0')
    plt.scatter(front1_1, front1_2, label='1')
    plt.scatter(front2_1, front2_2, label='2')
    plt.legend(loc=0)
    plt.show()


def plot_performance(best_T, best_fit_list, label, title):
    plt.plot(best_T, best_fit_list, label=label)
    plt.xlabel('evaluations')
    plt.ylabel('fitness')
    plt.title(title)
    plt.legend(loc='best')
    plt.plot(best_T[-1], best_fit_list[-1], 'ks')
    show_max = str(best_fit_list[-1])
    plt.annotate(show_max, xytext=(best_T[-1], best_fit_list[-1]),
                 xy=(best_T[-1], best_fit_list[-1]),
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))

    # plt.show()


def plot_performance_realtime(best_T, best_fit_list, label, title):
    plt.clf()  # 清除之前画的图
    plt.plot(best_T, best_fit_list, label=label)
    plt.xlabel('evaluations')
    plt.ylabel('fitness')
    plt.title(title)
    plt.legend(loc='best')
    plt.plot(best_T[-1], best_fit_list[-1], 'ks')
    show_max = str(best_fit_list[-1])
    plt.annotate(show_max, xytext=(best_T[-1], best_fit_list[-1]),
                 xy=(best_T[-1], best_fit_list[-1]),
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))
    plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
    plt.ioff()  # 关闭画图窗口
    # plt.show()
