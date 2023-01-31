# -*- coding: UTF-8 -*- #
"""
@filename:POSS.py
@author:201300086
@time:2023-01-30
"""
from tools.dataset import read_data_regression
from tools.my_operator import generate_binary, bit_wise_mutation, plot_performance_realtime, f1, f2, f3, f4, \
    bit_wise_mutation_cover, \
    record_time, clear_solution, plot_performance
import copy
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def is_better_regression(p, q):
    if f1(p) == f1(q) and f2(p) == f2(q):
        return 0
    elif f1(p) <= f1(q) and f2(p) <= f2(q):
        return 1
    elif f1(p) >= f1(q) and f2(p) >= f2(q):
        return -1
    else:
        return 10


def is_better_cover(p, q):
    if f3(p) == f3(q) and f4(p) == f4(q):
        return 0
    elif f3(p) >= f3(q) and f4(p) <= f4(q):
        return 1
    elif f3(p) <= f3(q) and f4(p) >= f4(q):
        return -1
    else:
        return 10


k = 8


@record_time
def POSS_regression(max_gen, times):
    # max_gen = 100
    pm = 0.05
    t = times
    path = "data/ionosphere.data"
    Z, DATA, Alpha = read_data_regression(path)

    # times=10
    fitnesses = [0 for i in range(max_gen)]
    evaluations = [i for i in range(max_gen)]
    while (times > 0):
        solution_group = [generate_binary(len(Alpha))]
        current_gen = 0
        best_fitness = 0.7
        fitnesses_tem = [0 for i in range(max_gen)]

        while (current_gen < max_gen):
            solution_group = clear_solution(solution_group, "regression")
            select_index = random.randint(0, len(solution_group) - 1)
            # mutation
            select_solution = bit_wise_mutation(solution_group[select_index], pm)
            # 遍历
            # 如果已有更优解，pass
            # 否则替换掉所有更差的
            better = 0
            new_group = []
            for i in range(len(solution_group)):
                if is_better_regression(select_solution, solution_group[i]) == -1:
                    better = 1
                    break
            if better == 0:
                # print("Generation number ", current_gen,f1(select_solution),f2(select_solution))
                new_group.append(select_solution)
                for i in range(len(solution_group)):
                    if is_better_regression(select_solution, solution_group[i]) == 10:  # 无法比较的
                        new_group.append(solution_group[i])
                # print("before",len(solution_group),f1(solution_group[0]))
                solution_group = copy.deepcopy(new_group)
                # print("after",len(solution_group),f1(solution_group[0]))
                if f2(select_solution) <= k and f1(select_solution) < best_fitness:
                    best_fitness = f1(select_solution)
                    print("Generation number ", current_gen, "at times ", t - times, best_fitness)
            fitnesses_tem[current_gen] += best_fitness / t
            current_gen += 1
        for i in range(max_gen):
            fitnesses[i] += fitnesses_tem[i]
        times -= 1
    plot_performance(evaluations, fitnesses, "POSS", "regression")
    # plt.show()


@record_time
def POSS_cover(max_gen, times):
    # max_gen = 2000
    pm = 0.003
    t = times
    fitnesses = [0 for i in range(max_gen)]
    evaluations = [i for i in range(max_gen)]
    while (times > 0):
        solution_group = [generate_binary(500)]
        current_gen = 0
        best_fitness = 0
        fitnesses_tem = [0 for i in range(max_gen)]
        while (current_gen < max_gen):
            solution_group = clear_solution(solution_group, "cover")
            select_index = random.randint(0, len(solution_group) - 1)
            # mutation
            select_solution = bit_wise_mutation_cover(solution_group[select_index], pm)
            # 遍历
            # 如果已有更优解，pass
            # 否则替换掉所有更差的
            better = 0
            new_group = []
            for i in range(len(solution_group)):
                if is_better_cover(select_solution, solution_group[i]) == -1:
                    better = 1
                    break
            if better == 0:
                # print("Generation number ", current_gen,best_fitness)#f3(select_solution),f4(select_solution),
                new_group.append(select_solution)
                for i in range(len(solution_group)):
                    if is_better_cover(select_solution, solution_group[i]) == 10:  # 无法比较的
                        new_group.append(solution_group[i])
                assert (len(new_group) <= len(solution_group) + 1)
                # print("before",len(solution_group),end=" ")
                solution_group = copy.deepcopy(new_group)
                # print("after",len(solution_group),end=" ")
                if f4(select_solution) <= k and f3(select_solution) > best_fitness:
                    best_fitness = f3(select_solution)
                    print("Generation number ", current_gen, "at times ", t - times, best_fitness)
            fitnesses_tem[current_gen] += best_fitness / t
            current_gen += 1
        for i in range(max_gen):
            fitnesses[i] += fitnesses_tem[i]
        times -= 1
    plot_performance(evaluations, fitnesses, "POSS", "maxcover")
    # plt.show()
# POSS_regression(3000,10)
# POSS_cover(60,10)
# plt.show()
