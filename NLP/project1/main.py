# -*- coding: UTF-8 -*- #
"""
@filename:main.py
@author:201300086
@time:2023-03-28
"""

from dataset import get_pretrain_features, load_train, load_test, test_acc, write_f, preprocess_keywords
from model import Multinomial_TF, Ensemble

if __name__ == "__main__":
    # 读取原始数据集
    textline = load_test("test.txt")
    train_X, train_Y = load_train("train.txt")

    # 分词等预处理
    train_X = preprocess_keywords(train_X)
    textline = preprocess_keywords(textline)

    # 集成学习
    pred, train_pred = Ensemble(train_X, train_Y, textline)

    # 打印训练集准确率
    print("训练集准确率:", test_acc(train_Y, train_pred))

    # 输出预测文件
    write_f(pred)
