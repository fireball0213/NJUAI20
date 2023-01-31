# -*- coding: UTF-8 -*- #
"""
@filename:MOEAD.py
@author:201300086
@time:2023-01-30
"""
import numpy as np
import copy
import math
max_gen = 200  # 迭代次数
population = 100
group = 200  # 存档数目

pops = []
EPs = []
nObj = 2
theta = 0.1  # 变异的概率
nVar = 30  # 决策变量个数
varMax = np.ones(nVar)
varMin = np.zeros(nVar)
yita1 = 2  # 交叉参数
yita2 = 5  # 变异参数
Lambda = []  # 权重向量


class pop():
    def __init__(self, var, cost):
        self.var = var
        self.cost = cost
        self.dominate = False

def ZDT2(x):
    return sum(x)
def ZDT3(x):
    return len(x)-sum(x)

def isDominates(x, y):
    # x是否支配y
    return (x <= y).all() and (x < y).any()


def determinDomination(p):
    for i in range(len(p)):
        p[i].dominate = False
    for i in range(0, len(pops) - 1):
        for j in range(i + 1, len(p)):
            if isDominates(p[i].cost, p[j].cost):
                p[j].dominate = True  # j被i支配
            if isDominates(p[j].cost, p[i].cost):
                p[i].dominate = True


T = max(math.ceil(0.15 * population), 2)
T = min(T, 15)  # 邻居的数目


def genVector2(nObj, npop, T):
    Lambda = []
    dist = np.zeros((npop, npop))
    for i in range(npop):
        w = np.random.rand(nObj)
        w = w / np.linalg.norm(w)
        #         w=w/sum(w)
        Lambda.append(w)
    for i in range(npop - 1):
        for j in range(i + 1, npop):
            dist[i][j] = dist[j][i] = np.linalg.norm(Lambda[i] - Lambda[j])
    sp_neighbors = np.argsort(dist, axis=1)
    sp_neighbors = sp_neighbors[:, :T]
    return Lambda, sp_neighbors


Lambda, sp_neighbors = genVector2(nObj, population, T)


def initPop(npop, nVar, varMin, varMax, fitness):
    global pops
    pops = []
    for i in range(npop):
        var = varMin + np.random.random(nVar) * (varMax - varMin)
        cost = fitness(var)
        pops.append(pop(var, cost))


initPop(population, nVar, varMin, varMax, ZDT2)

z = pops[0].cost
for p in range(population):
    for j in range(nObj):
        z[j] = min(pops[p].cost[j], z[j])
z = np.array(z)
determinDomination(pops)
EPs = copy.deepcopy([x for x in pops if x.dominate != True])


def cross_mutation2(p1, p2):
    gamma = 0.1  # 这个方式的效果很差
    alpha = -gamma + np.random.random(nVar) * (1 + 2 * gamma)
    y = alpha * p1 + (1 - alpha) * p2
    return y


def cross_mutation(p1, p2):
    # 交叉变异,不拷贝的话原始数据也变了
    y1 = np.copy(p1)
    y2 = np.copy(p2)
    y1, y2 = crossover(y1, y2)
    if np.random.rand() < theta:
        mutate(y1)
    if np.random.rand() < theta:
        mutate(y2)
    return y1, y2


def crossover(p1, p2):
    nVar = len(p1)
    gamma = 0
    for i in range(nVar):
        uj = np.random.rand()
        if uj <= 0.5:
            gamma = (2 * uj) ** (1 / (yita1 + 1))
        else:
            gamma = (1 / (2 * (1 - uj))) ** (1 / (yita1 + 1))
        p1[i] = 0.5 * ((1 + gamma) * p1[i] + (1 - gamma) * p2[i])
        p2[i] = 0.5 * ((1 - gamma) * p1[i] + (1 + gamma) * p2[i])
        p1[i] = min(p1[i], varMax[i])
        p1[i] = max(p1[i], varMin[i])
        p2[i] = min(p2[i], varMax[i])
        p2[i] = max(p2[i], varMin[i])
    return p1, p2


def mutate(p):
    # 用的是多项式变异，对每个决策变量进行变异
    dj = 0
    for i in range(len(p)):
        uj = np.random.rand()
        if uj < 0.5:
            dj = (2 * uj) ** (1 / (yita2 + 1)) - 1
        else:
            dj = 1 - (2 * (1 - uj)) ** (1 / (yita2 + 1))
        p[i] = p[i] + dj
        p[i] = min(p[i], varMax[i])
        p[i] = max(p[i], varMin[i])


def generate_next(idx, xk, xl, fitness):
    y0, y1 = cross_mutation(xk, xl)
    #     y0=cross_mutation2(xk,xl)
    # 对y进行修复根据约束
    for i in range(nVar):
        y0[i] = max(varMin[i], y0[i])
        y0[i] = min(varMax[i], y0[i])
        #     return y0
        y1[i] = max(varMin[i], y1[i])
        y1[i] = min(varMax[i], y1[i])
    fx1 = np.array(fitness(y0))
    fx2 = np.array(fitness(y1))
    if isDominates(fx1, fx2):
        return y0
    elif isDominates(fx2, fx1):
        return y1
    else:
        if np.random.rand() < 0.5:
            return y0
        else:
            return y1

#更新邻域解
def update_neighbor(idx, y):
    # 若gy<gx更新,用的权重是邻居的权重
    Bi = sp_neighbors[idx]
    fy = y.cost
    for j in range(len(Bi)):
        w = Lambda[Bi[j]]
        maxn_y = max(w * abs(fy - z))
        maxn_x = max(w * abs(pops[Bi[j]].cost - z))
        if maxn_x >= maxn_y:
            pops[Bi[j]] = y



def MOEAD():
    #主循环
    for j in range(max_gen):
        if j % 10 == 0:
            print("=" * 10, j, "=" * 10)
            print(z, len(EPs))
        for i in range(population):
            Bi = sp_neighbors[i]
            choice = np.random.choice(T, 2, replace=False)  # 选出来的邻居应该不重复
            k = Bi[choice[0]]
            l = Bi[choice[1]]
            xk = pops[k]
            xl = pops[l]
            # 产生新的解，并对解进行修复
            y = generate_next(i, xk.var, xl.var, ZDT3)
            fv_y = np.array(ZDT3(y))
            y = pop(y, fv_y)
            # 更新z,
            t = z > fv_y
            z[t] = fv_y[t]
            # 更新邻域解
            update_neighbor(i, y)
            ep = False
            delete = []
            for k in range(len(EPs)):
                if (fv_y == EPs[k].cost).all():  # 如果有一样的就不用算了啊
                    ep = True
                    break
                if isDominates(fv_y, EPs[k].cost):
                    delete.append(EPs[k])
                elif ep == False and isDominates(EPs[k].cost, fv_y):
                    ep = True
                    break  # 后面就不用看了，最好也是互不支配
            if len(delete) != 0:
                for k in range(len(delete)):
                    EPs.remove(delete[k])
            if ep == False:
                EPs.append(y)
            while len(EPs) > group:
                select = np.random.randint(0, len(EPs))
                del EPs[select]

