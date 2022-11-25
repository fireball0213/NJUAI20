import networkx as nx
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import random

matplotlib.use('TkAgg')


def generate_regular_graph(args):
    # 各顶点的度均相同的无向简单图
    # 这里简单以正则图为例, 鼓励同学们尝试在其他类型的图(具体可查看如下的nx文档)上测试算法性能
    # nx文档 https://networkx.org/documentation/stable/reference/generators.html
    graph = nx.random_graphs.random_regular_graph(d=args.n_d, n=args.n_nodes, seed=args.seed_g)
    return graph, len(graph.nodes), len(graph.edges)


def generate_gset_graph(args):
    # 这里提供了比较流行的图集合: Gset, 用于进行分割
    dir = './Gset/'
    fname = dir + 'G' + str(args.gset_id) + '.txt'
    graph_file = open(fname)
    n_nodes, n_e = graph_file.readline().rstrip().split(' ')
    # print(n_nodes, n_e)
    nodes = [i for i in range(int(n_nodes))]
    edges = []
    for line in graph_file:
        n1, n2, w = line.split(' ')
        edges.append((int(n1) - 1, int(n2) - 1, int(w)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return graph, len(graph.nodes), len(graph.edges)


def graph_generator(args):  # 生成图
    if args.graph_type == 'regular':
        return generate_regular_graph(args)
    elif args.graph_type == 'gset':
        return generate_gset_graph(args)
    else:
        raise NotImplementedError(f'Wrong graph_tpye')
