# -*- coding = utf-8 -*-
# Time : 2022/6/5 20:18
# Author : 201300086史浩男
# File : Iris-Bayes.py
# Software : PyCharm
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def load_data():
    # 以feature , label的 形 式 返 回 数 据 集
    feature, label = load_iris(return_X_y=True)
    print(feature.shape)  # (150, 4)
    print(label.shape)  # (150,)
    return feature, label


def split_data(feature, label):
    feature_train, feature_test, label_train, label_test = \
        train_test_split(feature, label, test_size=0.2, random_state=0)
    return feature_train, feature_test, label_train, label_test


feature, label = load_data()
feature_train, feature_test, label_train, label_test = split_data(feature, label)
# print(feature_train, feature_test, label_train, label_test)
print('Counter(label_train):', Counter(label_train))
print('Counter(label_test):', Counter(label_test))
print(Counter(label_train)[1])

clf = GaussianNB()
clf = clf.fit(feature_train, label_train)
clf.class_prior_ = [1. / 3., 1. / 3., 1. / 3.]  # 分数的表示方式
# clf.class_count_ = [39,37, 44]#每个类别的样本书，这个指定后每个类别概率也自动指定了
print(clf.class_prior_)
print("==Predict result in GaussianNB by predict==")
print(clf.predict(feature_test))
print(clf.predict_proba(feature_test))
print(clf.score(feature_test, label_test))

plt.subplots_adjust(wspace=0.4, hspace=0.8)
for i in range(3):
    for j in range(4):
        # 第 (i, j) 张图
        plt.subplot(3, 4, i * 4 + j + 1)
        plt.hist(feature_train[label_train == i][:, j], color='purple')
        plt.title(f"P(x_{j + 1}|y={i})")
plt.show()
# print(clf.predict_proba(np.array([[5.8,2.8,5.1,2.4]])))
# print(clf.predict_log_proba(np.array([[5.8,2.8,5.1,2.4]])))
