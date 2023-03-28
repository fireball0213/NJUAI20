# -*- coding: UTF-8 -*- #
"""
@filename:model.py
@author:201300086
@time:2023-03-28
"""
from dataset import preprocess_keywords,get_features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import torch


#

#朴素贝叶斯：0.66
def Multinomial(train_data,train_labels,textwords):
    _,_, train_features = get_features(train_data)
    train_features = (train_features - train_features.min()) / (train_features.max() - train_features.min())
    # vectorizer = TfidfVectorizer()
    # train_features = vectorizer.fit_transform(train_data)
    # 训练朴素贝叶斯分类器
    clf = MultinomialNB()
    clf.fit(train_features, train_labels)
    # preds=[]
    # train_pred=[]
    _, _, test_features = get_features(textwords)
    test_features = (test_features - test_features.min()) / (test_features.max() - test_features.min())
    preds=clf.predict(test_features)
    # for i in range(len(textwords)):
    #     # 将文本转换为tf-idf特征向量
    #     #test_features = vectorizer.transform([' '.join(textwords[i])])
    #     # 预测情感
    #     pred = clf.predict(test_features[i].reshape(1,-1))[0]
    #     preds.append(pred)
    train_pred=clf.predict(train_features)
    # for i in range(len(train_data)):
    #     # 预测情感
    #     pred = clf.predict(train_features[i])[0]
    #     train_pred.append(pred)
    print(len(preds))
    print(len(train_pred))
    return preds,train_pred

def Ensemble(train_data,train_labels,textwords):
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_data)

    # 训练朴素贝叶斯分类器
    clf = MultinomialNB()
    clf.fit(train_features, train_labels)
    preds = []
    train_pred = []
    for i in range(len(textwords)):
        # 将文本转换为tf-idf特征向量
        features = vectorizer.transform([' '.join(textwords[i])])
        # 预测情感
        pred = clf.predict(features)[0]
        preds.append(pred)
    # for i in range(len(train_data)):
    #     # 将文本转换为tf-idf特征向量
    #     features = vectorizer.transform([' '.join(train_data[i])])
    #     # 预测情感
    #     pred = clf.predict(features)[0]
    #     train_pred.append(pred)
    return preds, train_pred