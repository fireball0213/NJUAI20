# -*- coding: UTF-8 -*- #
"""
@filename:NSGA_II.py
@author:201300086
@time:2023-01-29
"""
from tools.dataset import read_data_regression
from tools.my_operator import generate_binary, bit_wise_mutation, one_point_crossover, plot_performance, \
    f3, f4, bit_wise_mutation_cover, \
    record_time, clear_solution, f1, f2, plot_performance_realtime
import math
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def generate_offsprings(solution_group, front, pm, pc, problem):
    parent_index = []
    if len(front) >= 5:
        parent_index = front[0][:] + front[1][:] + front[2] + front[3] + front[4]
    else:
        for i in front:
            parent_index += i
    parents = [solution_group[i] for i in parent_index]
    offsprings = []
    if problem == "regression":
        for i in range(len(parents)):
            for j in range(i + 1, 2 * len(parents)):
                jj = j % len(parents)
                if f1(parents[i]) != f1(parents[jj]):
                    tem1 = bit_wise_mutation(parents[i], pm)
                    tem2 = bit_wise_mutation(parents[jj], pm)
                    off1, off2 = one_point_crossover(tem1, tem2, pc)
                    if len(offsprings) < 2 * Group_size:
                        offsprings.append(off1)
                    if len(offsprings) < 2 * Group_size:
                        offsprings.append(off2)
                    break
                else:
                    pass
    if problem == "cover":
        for i in range(len(parents)):
            for j in range(i + 1, 2 * len(parents)):
                jj = j % len(parents)
                if f3(parents[i]) != f3(parents[jj]):
                    tem1 = bit_wise_mutation_cover(parents[i], pm)
                    tem2 = bit_wise_mutation_cover(parents[jj], pm)
                    off1, off2 = one_point_crossover(tem1, tem2, pc)
                    if len(offsprings) < 2 * Group_size:
                        offsprings.append(off1)
                    if len(offsprings) < 2 * Group_size:
                        offsprings.append(off2)
                    break
                else:
                    pass
    return offsprings


def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf  # 最小值替换
    return sorted_list  # 存储index排序


# Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2, problem):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []  # 比当前解差的集合
        n[p] = 0  # 比当前解好的个数
        for q in range(0, len(values1)):
            if problem == "regression":
                if values1[q] == values1[p] and values2[q] == values2[p]:
                    pass
                elif values1[q] <= values1[p] and values2[q] <= values2[p]:
                    n[p] = n[p] + 1
                elif values1[q] >= values1[p] and values2[q] >= values2[p]:
                    if q not in S[p]:
                        S[p].append(q)  # q比p差
            elif problem == "cover":
                if values1[q] == values1[p] and values2[q] == values2[p]:
                    pass
                elif values1[q] >= values1[p] and values2[q] <= values2[p]:
                    n[p] = n[p] + 1
                elif values1[q] <= values1[p] and values2[q] >= values2[p]:
                    if q not in S[p]:
                        S[p].append(q)  # q比p差
        if n[p] == 0:
            rank[p] = 0  # 最优
            if p not in front[0]:  # 帕累托前沿
                front[0].append(p)

    i = 0
    while (front[i] != []):  # 得出其他前沿
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    # del front[0]
    return front


# Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + \
                      (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (max(values1) - min(values1)) + \
                      (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))
    return distance


k = 8
Group_size = 8


@record_time
def NSGA_cover(max_gen, times):
    # max_gen = 500
    pm = 0.008
    pc = 1
    t = times
    # 初始化种群
    fitnesses = [0 for i in range(max_gen)]
    evaluations = [i for i in range(max_gen)]
    while (times > 0):
        solution_group = [generate_binary(500, "cover") for i in range(0, Group_size)]
        current_gen = 0
        best_fitness = 0
        fitnesses_tem = [0 for i in range(max_gen)]
        # 主循环
        while (current_gen < max_gen):
            assert len(solution_group) > 0
            solution_group = clear_solution(solution_group, "cover")
            assert len(solution_group) > 0
            f3_values = [f3(solution_group[i]) for i in range(0, len(solution_group))]
            f4_values = [f4(solution_group[i]) for i in range(0, len(solution_group))]
            front1 = fast_non_dominated_sort(f3_values[:], f4_values[:], "cover")
            # print("The best fitness within k :", best_fitness, "in generation ", current_gen)
            # 检测当前最好解
            for v in front1[0]:
                a = solution_group[v]
                if f4(a) <= 8 and f3(a) > best_fitness:
                    best_fitness = f3(a)
                    print("Generation number ", current_gen, "at times ", t - times, best_fitness)
                    # print("The best fitness within k :", best_fitness, "in generation ", current_gen)

            distance = []
            for i in range(0, len(front1)):
                distance.append(crowding_distance(f3_values[:], f4_values[:], front1[i][:]))

            # Generating offsprings
            solution_group2 = generate_offsprings(solution_group, front1, pm, pc, "cover")

            f3_values2 = [f3(solution_group2[i]) for i in range(0, len(solution_group2))]
            f4_values2 = [f4(solution_group2[i]) for i in range(0, len(solution_group2))]
            front2 = fast_non_dominated_sort(f3_values2[:], f4_values2[:], "cover")
            distance2 = []
            for i in range(0, len(front2)):
                distance2.append(crowding_distance(f3_values2[:], f4_values2[:], front2[i][:]))

            # 优秀父辈直接继承
            old_solution = []
            for k in range(0, 1):
                for v in front1[k]:
                    old_solution.append(v)
                    # print(f2_values[v], end=" ")

            solution_group = [solution_group[i] for i in old_solution]
            assert len(solution_group) > 0

            # 选拔优秀子代
            new_solution = []
            for i in range(0, len(front2)):  # 遍历每个前沿
                non_dominated_sorted_solution2_1 = \
                    [index_of(front2[i][j], front2[i]) for j in range(0, len(front2[i]))]
                front22 = sort_by_values(non_dominated_sorted_solution2_1[:], distance2[i][:])  # 每个前沿内用distance排序
                front = [front2[i][front22[j]] for j in range(0, len(front2[i]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if (len(new_solution) == Group_size - len(old_solution)):
                        break
                if (len(new_solution) == Group_size - len(old_solution)):
                    break
            solution_group += [solution_group2[i] for i in new_solution]  # 每轮迭代更新的是solution
            fitnesses_tem[current_gen] += best_fitness / t
            current_gen += 1
        for i in range(max_gen):
            fitnesses[i] += fitnesses_tem[i]
        times -= 1
    plot_performance(evaluations, fitnesses, "NSGA-II", "maxcover")
    # plt.show()


@record_time
def NSGA_regression(max_gen, times):
    # max_gen = 100
    pm = 0.08
    pc = 1
    t = times
    path = "data/ionosphere.data"
    Z, DATA, Alpha = read_data_regression(path)

    fitnesses = [0 for i in range(max_gen)]
    evaluations = [i for i in range(max_gen)]
    while (times > 0):
        # 初始化种群
        solution_group = [generate_binary(len(Alpha)) for i in range(0, Group_size)]
        current_gen = 0
        best_fitness = 0.7
        fitnesses_tem = [0 for i in range(max_gen)]

        # 主循环
        while (current_gen < max_gen):
            assert len(solution_group) > 0
            solution_group = clear_solution(solution_group, "regression")
            assert len(solution_group) > 0
            f1_values = [f1(solution_group[i]) for i in range(0, len(solution_group))]
            f2_values = [f2(solution_group[i]) for i in range(0, len(solution_group))]
            front1 = fast_non_dominated_sort(f1_values[:], f2_values[:], "regression")

            for v in front1[0]:
                a = solution_group[v]
                if f2(a) <= 8 and f1(a) < best_fitness:
                    best_fitness = f1(a)
                    print("Generation number ", current_gen, "at times ", t - times, best_fitness)
                    # print("The best fitness within k :", best_fitness, "in generation ", current_gen)

            distance = []
            for i in range(0, len(front1)):
                distance.append(crowding_distance(f1_values[:], f2_values[:], front1[i][:]))

            # Generating offsprings
            solution_group2 = generate_offsprings(solution_group, front1, pm, pc, "regression")

            function1_values2 = [f1(solution_group2[i]) for i in range(0, len(solution_group2))]
            function2_values2 = [f2(solution_group2[i]) for i in range(0, len(solution_group2))]
            front2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:], "regression")
            distance2 = []
            for i in range(0, len(front2)):
                distance2.append(crowding_distance(function1_values2[:], function2_values2[:], front2[i][:]))

            # 优秀父辈直接继承
            old_solution = []
            for k in range(0, 1):
                for v in front1[k]:
                    old_solution.append(v)
                    # print(f2_values[v], end=" ")

            solution_group = [solution_group[i] for i in old_solution]
            # print(111111,len(solution_group))
            assert len(solution_group) > 0
            # print("old",len(old_solution))

            # 选拔优秀子代
            new_solution = []
            for i in range(0, len(front2)):  # 遍历每个前沿
                non_dominated_sorted_solution2_1 = \
                    [index_of(front2[i][j], front2[i]) for j in range(0, len(front2[i]))]
                front22 = sort_by_values(non_dominated_sorted_solution2_1[:], distance2[i][:])  # 每个前沿内用distance排序
                front = [front2[i][front22[j]] for j in range(0, len(front2[i]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if (len(new_solution) == Group_size - len(old_solution)):
                        break
                if (len(new_solution) == Group_size - len(old_solution)):
                    break
            solution_group += [solution_group2[i] for i in new_solution]  # 每轮迭代更新的是solution

            fitnesses_tem[current_gen] += best_fitness / t
            current_gen += 1
        for i in range(max_gen):
            fitnesses[i] += fitnesses_tem[i]
        times -= 1
    plot_performance(evaluations, fitnesses, "NSGA-II", "regression")


@record_time
def MOEAD_cover(max_gen, times):
    # max_gen = 500
    pm = 0.005
    pc = 1
    t = times
    # 初始化种群
    fitnesses = [0 for i in range(max_gen)]
    evaluations = [i for i in range(max_gen)]
    while (times > 0):
        solution_group = [generate_binary(500, "cover") for i in range(0, Group_size)]
        current_gen = 0
        best_fitness = 0
        fitnesses_tem = [0 for i in range(max_gen)]
        # 主循环
        while (current_gen < max_gen):
            assert len(solution_group) > 0
            solution_group = clear_solution(solution_group, "cover")
            assert len(solution_group) > 0
            f3_values = [f3(solution_group[i]) for i in range(0, len(solution_group))]
            f4_values = [f4(solution_group[i]) for i in range(0, len(solution_group))]
            front1 = fast_non_dominated_sort(f3_values[:], f4_values[:], "cover")
            # print("The best fitness within k :", best_fitness, "in generation ", current_gen)
            # 检测当前最好解
            for v in front1[0]:
                a = solution_group[v]
                if f4(a) <= 8 and f3(a) > best_fitness:
                    best_fitness = f3(a)
                    print("Generation number ", current_gen, "at times ", t - times, best_fitness)
                    # print("The best fitness within k :", best_fitness, "in generation ", current_gen)

            distance = []
            for i in range(0, len(front1)):
                distance.append(crowding_distance(f3_values[:], f4_values[:], front1[i][:]))

            # Generating offsprings
            solution_group2 = generate_offsprings(solution_group, front1, pm, pc, "cover")

            f3_values2 = [f3(solution_group2[i]) for i in range(0, len(solution_group2))]
            f4_values2 = [f4(solution_group2[i]) for i in range(0, len(solution_group2))]
            front2 = fast_non_dominated_sort(f3_values2[:], f4_values2[:], "cover")
            distance2 = []
            for i in range(0, len(front2)):
                distance2.append(crowding_distance(f3_values2[:], f4_values2[:], front2[i][:]))

            # 优秀父辈直接继承
            old_solution = []
            for k in range(0, len(front1)):
                for v in front1[k]:
                    if len(old_solution)<Group_size/2:
                        old_solution.append(v)
                    # print(f2_values[v], end=" ")

            solution_group = [solution_group[i] for i in old_solution]
            assert len(solution_group) > 0

            # 选拔优秀子代
            new_solution = []
            for i in range(0, len(front2)):  # 遍历每个前沿
                non_dominated_sorted_solution2_1 = \
                    [index_of(front2[i][j], front2[i]) for j in range(0, len(front2[i]))]
                front22 = sort_by_values(non_dominated_sorted_solution2_1[:], distance2[i][:])  # 每个前沿内用distance排序
                front = [front2[i][front22[j]] for j in range(0, len(front2[i]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if (len(new_solution) == Group_size - len(old_solution)):
                        break
                if (len(new_solution) == Group_size - len(old_solution)):
                    break
            solution_group += [solution_group2[i] for i in new_solution]  # 每轮迭代更新的是solution
            fitnesses_tem[current_gen] += best_fitness / t
            current_gen += 1
        for i in range(max_gen):
            fitnesses[i] += fitnesses_tem[i]
        times -= 1
    plot_performance(evaluations, fitnesses, "MOEAD", "maxcover")
    # plt.show()


@record_time
def MOEAD_regression(max_gen, times):
    # max_gen = 100
    pm = 0.1
    pc = 1
    t = times
    path = "data/ionosphere.data"
    Z, DATA, Alpha = read_data_regression(path)

    fitnesses = [0 for i in range(max_gen)]
    evaluations = [i for i in range(max_gen)]
    while (times > 0):
        # 初始化种群
        solution_group = [generate_binary(len(Alpha)) for i in range(0, Group_size)]
        current_gen = 0
        best_fitness = 0.7
        fitnesses_tem = [0 for i in range(max_gen)]

        # 主循环
        while (current_gen < max_gen):
            assert len(solution_group) > 0
            solution_group = clear_solution(solution_group, "regression")
            assert len(solution_group) > 0
            f1_values = [f1(solution_group[i]) for i in range(0, len(solution_group))]
            f2_values = [f2(solution_group[i]) for i in range(0, len(solution_group))]
            front1 = fast_non_dominated_sort(f1_values[:], f2_values[:], "regression")

            for v in front1[0]:
                a = solution_group[v]
                if f2(a) <= 8 and f1(a) < best_fitness:
                    best_fitness = f1(a)
                    print("Generation number ", current_gen, "at times ", t - times, best_fitness)
                    # print("The best fitness within k :", best_fitness, "in generation ", current_gen)

            distance = []
            for i in range(0, len(front1)):
                distance.append(crowding_distance(f1_values[:], f2_values[:], front1[i][:]))

            # Generating offsprings
            solution_group2 = generate_offsprings(solution_group, front1, pm, pc, "regression")

            function1_values2 = [f1(solution_group2[i]) for i in range(0, len(solution_group2))]
            function2_values2 = [f2(solution_group2[i]) for i in range(0, len(solution_group2))]
            front2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:], "regression")
            distance2 = []
            for i in range(0, len(front2)):
                distance2.append(crowding_distance(function1_values2[:], function2_values2[:], front2[i][:]))

            # 优秀父辈直接继承
            old_solution = []
            for k in range(0, 1):
                for v in front1[k]:
                    old_solution.append(v)
                    # print(f2_values[v], end=" ")

            solution_group = [solution_group[i] for i in old_solution]
            # print(111111,len(solution_group))
            assert len(solution_group) > 0
            # print("old",len(old_solution))

            # 选拔优秀子代
            new_solution = []
            for i in range(0, len(front2)):  # 遍历每个前沿
                non_dominated_sorted_solution2_1 = \
                    [index_of(front2[i][j], front2[i]) for j in range(0, len(front2[i]))]
                front22 = sort_by_values(non_dominated_sorted_solution2_1[:], distance2[i][:])  # 每个前沿内用distance排序
                front = [front2[i][front22[j]] for j in range(0, len(front2[i]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if (len(new_solution) == Group_size - len(old_solution)):
                        break
                if (len(new_solution) == Group_size - len(old_solution)):
                    break
            solution_group += [solution_group2[i] for i in new_solution]  # 每轮迭代更新的是solution

            fitnesses_tem[current_gen] += best_fitness / t
            current_gen += 1
        for i in range(max_gen):
            fitnesses[i] += fitnesses_tem[i]
        times -= 1
    plot_performance(evaluations, fitnesses, "MOEAD", "regression")
# NSGA_regression(400,10)
# NSGA_cover(600,10)
# plt.show()
