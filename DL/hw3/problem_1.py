# -*- coding: UTF-8 -*- #
"""
@filename:problem_1.py
@author:201300086
@time:2022-12-16
"""
import torch


# 测试flow
# a=torch.linspace(0, 100, steps=10, out=None)
# b=torch.linspace(0, 1e-45, steps=10, out=None)
# print("截断前：")
# print(torch.exp(a))
# print(torch.log(b))
# a = torch.clamp(a, min=1e-8, max=80)
# b = torch.clamp(b, min=1e-8, max=80)
# print("截断后：")
# print(torch.exp(a))
# print(torch.log(b))
# print(torch.exp(a).isinf().long().sum())

def my_func(a, bias):
    M = torch.max(a)
    exp = torch.exp(a - M)
    log = torch.log(bias + torch.sum(exp))
    print("+inf:", exp.isinf().long().sum().item(), end="  ")  # 统计+inf个数
    print("-inf:", log.isinf().long().sum().item(), end="  ")  # 统计-inf个数
    result = (M + log).item()
    print("result=", result)


def old_func(a):
    exp = torch.exp(a)
    log = torch.log(torch.sum(exp))
    print("+inf:", exp.isinf().long().sum().item(), end="  ")  # 统计+inf个数
    print("-inf:", log.isinf().long().sum().item(), end="  ")  # 统计-inf个数
    result = log.item()
    print("result=", result)


tf = []
with open(file="test_case.txt", mode="r", encoding="utf-8") as f:
    data = f.read()
    data = data.split('\n')
    for i in range(len(data)):
        tem = data[i].split()
        tem = list(map(int, tem))
        tf.append(tem)
tf.pop()
for i in range(len(tf)):
    my_func(torch.Tensor(tf[i]), 1e-8)
for i in range(len(tf)):
    old_func(torch.Tensor(tf[i]))
