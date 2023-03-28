# -*- coding: UTF-8 -*- #
"""
@filename:main.py
@author:201300086
@time:2023-03-28
"""

from dataset import *
from model import Multinomial

textline = load_test("test.txt")
train_X,train_Y=load_train("train.txt")
# print(train_X[0])
train_X=preprocess_keywords(train_X)
# print(train_X[0])

textwords=preprocess_keywords(textline)

# for i in range(10):
#     print(textline[i])
#     print(preprocess(textline[i]))
#     print(train_Y[i])
#     print(preprocess(train_X[i]))

pred,train_pred=Multinomial(train_X,train_Y,textwords)




# 定义Embedding层
#embedding = torch.nn.Embedding.from_pretrained(pretrained_weights)
print(test_acc(train_Y,train_pred))
write_f(pred)