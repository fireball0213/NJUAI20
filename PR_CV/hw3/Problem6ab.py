# -*- coding: UTF-8 -*- #
"""
@filename:Problem6ab.py
@author:201300086
@time:2023-05-09
"""

import heapq
from collections import defaultdict
import numpy as np

#一个计算信息熵的函数，输入为一个列表
import math

#一个计算信息熵的函数，输入为一个概率分布的列表
def entropy(list):
    entropy = 0
    for i in list:
        entropy += -i*math.log(i,2)
    return entropy

class Node:
    def __init__(self, value, freq, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def quantize(data_list, num_levels):
    # 量化数据到给定的级别
    min_val, max_val = min(data_list), max(data_list)
    levels = np.linspace(min_val, max_val, num_levels)
    quantized_data = np.digitize(data_list, levels)-1
    return quantized_data, levels


def huffman_encoding(data_list, prob_list):
    # 构造频率字典
    freq_dict = defaultdict(int)
    for data, prob in zip(data_list, prob_list):
        freq_dict[data] = prob

    # 构造优先队列
    priority_queue = [Node(value, freq) for value, freq in freq_dict.items()]
    heapq.heapify(priority_queue)

    # 构造哈夫曼树
    while len(priority_queue) != 1:
        node1 = heapq.heappop(priority_queue)
        node2 = heapq.heappop(priority_queue)
        merged = Node(None, node1.freq + node2.freq, node1, node2)
        heapq.heappush(priority_queue, merged)

    # 生成哈夫曼编码
    root = priority_queue[0]
    huffman_code = {}
    stack = [(root, "")]
    while stack:
        node, code = stack.pop()
        if node is not None:
            if node.value is not None:
                huffman_code[node.value] = code
            stack.append((node.left, code + "0"))
            stack.append((node.right, code + "1"))

    # 计算码率
    bitrate = sum([freq_dict[data] * len(huffman_code[data]) for data in data_list])

    # 返回哈夫曼编码字典和码率
    return huffman_code, bitrate


def huffman_decoding(huffman_code, encoded_data):
    # 构造哈夫曼编码到原始数据的映射
    code_to_data = {v: k for k, v in huffman_code.items()}

    # 解码
    decoded_data = [code_to_data[code] for code in encoded_data]

    return decoded_data

def dequantize(quantized_data, levels):
    # 将量化的数据映射回原始范围
    dequantized_data = [levels[i] for i in quantized_data]
    return dequantized_data

def mse(original_data, reconstructed_data):
    return ((np.array(original_data) - np.array(reconstructed_data))**2).mean()

#计算性能指标，公式为MSE+λ*bitrate，其中λ为函数输入
def performance(original_data, reconstructed_data, bitrate, lamda):
    return mse(original_data, reconstructed_data) + lamda*bitrate



#主函数
if __name__=='__main__':

    #6.a
    lst_x=[0.5,0.25,0.25]
    lst_y=[0.5,0.5]
    lst_x_=[0.5,0.5]
    print(entropy(lst_x),entropy(lst_y),entropy(lst_x_))

    #6.b
    num_levels = 5
    data_list = [1, 2, 3]
    prob_list = [0.5, 0.25, 0.25]
    quantized_data, levels = quantize(data_list, num_levels)
    huffman_code, bitrate = huffman_encoding(quantized_data, prob_list)
    encoded_data = [huffman_code[data] for data in quantized_data]
    decoded_data = huffman_decoding(huffman_code, encoded_data)
    reconstructed_data = dequantize(decoded_data, levels)
    Mse = mse(data_list, reconstructed_data)
    print(f'Original data: {data_list}',f'  prob_list: {prob_list}',)
    print(f'Encoded data: {encoded_data}')
    print(f'Bitrate: {bitrate} bits/symbol')
    print(f'Reconstructed data: {reconstructed_data}')
    print(f'MSE: {Mse}')
    print(f'Performance when λ = 0.1 : {performance(data_list, reconstructed_data, bitrate, 0.1)}')
    print(f'Performance when λ = 1 : {performance(data_list, reconstructed_data, bitrate, 1)}')
    print(f'Performance when λ = 10 : {performance(data_list, reconstructed_data, bitrate, 10)}')



