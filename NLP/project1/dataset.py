# -*- coding: UTF-8 -*- #
"""
@filename:dataset.py
@author:201300086
@time:2023-03-27
"""
import numpy as np
import random
from collections import Counter
import time
import jieba
import opencc
import jieba.analyse
import re
from torchtext.vocab import Vectors
import numpy as np
import pandas as pd
import torch
def load_train(filename):
    with open(filename, encoding='utf-8') as f:
        text = f.readlines()
        # 创建简繁转换器对象
        converter = opencc.OpenCC('t2s.json')
        # 将繁体中文文本转换为简体中文文本
        X= []
        Y=[]
        for i in range(len(text)):
            Y.append(text[i][0])
            X.append(converter.convert(text[i][2:]))
    return X,Y
def load_test(filename):
    with open(filename, encoding='utf-8') as f:
        text = f.readlines()
        # 创建简繁转换器对象
        converter = opencc.OpenCC('t2s.json')
        # 将繁体中文文本转换为简体中文文本
        textline= []
        for i in range(len(text)):
            textline.append(converter.convert(text[i]))
    return textline


def write_f(label):
    with open('201300086.txt', 'w', encoding='utf-8') as f:
        for i in range(len(label)):
            f.write(label[i] + '\n')

def test_acc(Y,pred):
    acc=0
    max=len(Y)
    for i in range(max):
        acc+=Y[i]==pred[i]
    return acc/max


def preprocess_keywords(text):
    words=[]
    for textline in text:
        # 去除特殊符号
        pattern = '[^\u4e00-\u9fa5a-zA-Z0-9]+'
        textline = re.sub(pattern, '', textline)

        # 使用jieba进行分词
        seg_list = list(jieba.cut(textline, cut_all=False))
        # 去除停用词
        stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()]
        seg_list = [word for word in seg_list if word not in stopwords]
        # 提取关键词
        keywords = jieba.analyse.extract_tags(textline, topK=20, withWeight=True, allowPOS=())
        # 将分词结果和关键词组合成一个列表
        word = list(seg_list) + [word[0] for word in keywords]
        words.append(word)
    return words

def preprocess_vec():#加工预训练的词向量文件,输出文件
    textline = []
    title=[]
    with open("sgns.weibo.bigram-char", encoding='utf-8') as f:
        text = f.readlines()
        for i in range(len(text)):
            line = text[i].split()
            if len(line) == 301:
                title.append(line[0])
                textline.append(line)
    with open('sgns.weibo_vocab.txt', 'w', encoding='utf-8') as f:
        for i in range(len(title)):
                f.write(title[i] + "\n")
    with open('sgns.weibo.txt', 'w', encoding='utf-8') as f:
        for i in range(len(textline)):
            for j in range(len(textline[i])):
                f.write(str(textline[i][j]) + " ")
            f.write('\n')
def record_time(func):
    def wrapper(*args, **kwargs):  # 包装带参函数
        start_time = time.perf_counter()
        a = func(*args, **kwargs)  # 包装带参函数
        end_time = time.perf_counter()
        print('time=', end_time - start_time)
        return a  # 有返回值的函数必须给个返回

    return wrapper

@record_time
def get_features(train_X):#
    """
    :return: 词向量矩阵,预训练词向量(可索引)，转成feature的原分词后文本平均
    """
    # 定义词汇表
    vocab = pd.read_csv("sgns.weibo_vocab.txt", encoding='utf-8', sep=" ", header=None)
    vocab = list(vocab[0])

    # 加载预训练的词向量
    glove = Vectors(name='sgns.weibo.txt')
    print("预训练词向量加载完成")

    #转feature
    features=torch.zeros((len(train_X),300))
    for i in range(len(train_X)):
        feature_mean = torch.zeros(300)
        for j in range(len(train_X[i])):
            feature_mean+=glove[train_X[i][j]]
        if len(train_X[i])>0:
            feature_mean/=len(train_X[i])
        else:
            pass
        features[i]=feature_mean#.reshape(1,-1)
    # 构建词向量矩阵
    embedding_dim = 300
    pretrained_weights = torch.zeros(len(vocab), embedding_dim)
    for i, word in enumerate(vocab):
        try:
            pretrained_weights[i] = glove[word]
        except KeyError:
            pass
    #print("词向量矩阵加载完成")
    return pretrained_weights,glove,features

# if __name__ == "__main__":
