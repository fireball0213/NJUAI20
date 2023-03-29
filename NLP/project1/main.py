# -*- coding: UTF-8 -*- #
"""
@filename:main.py
@author:201300086
@time:2023-03-28
"""

from dataset import get_pretrain_features,load_train,load_test,test_acc,write_f,preprocess_keywords
from model import Multinomial,Ensemble

textline = load_test("test.txt")
train_X,train_Y=load_train("train.txt")
# print(train_X[0])
train_X=preprocess_keywords(train_X)
textline=preprocess_keywords(textline)

# for i in range(10):
#     print(textline[i])
#     print(preprocess(textline[i]))
#     print(train_Y[i])
#     print(preprocess(train_X[i]))

pred,train_pred=Multinomial(train_X,train_Y,textline)
#pred,train_pred=Ensemble(train_X,train_Y,textline)




# 定义Embedding层
#embedding = torch.nn.Embedding.from_pretrained(pretrained_weights)
print(test_acc(train_Y,train_pred))
write_f(pred)