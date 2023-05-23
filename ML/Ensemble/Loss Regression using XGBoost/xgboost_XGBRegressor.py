# -*- coding: UTF-8 -*- #

import math, copy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


import numpy as np
from sklearn.ensemble import  AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, \
    StackingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib
from sklearn.metrics import mean_squared_error
from my_plot import plot_error_estimator, plot_error_history, plot_features_importance, plot_pred

font = {'family': 'Times New Roman', 'size': 18}
matplotlib.rc('font', **font)
matplotlib.rcParams.update(
    {'font.family': 'Times New Roman', 'font.size': 18, })
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.sans-serif'] = ['SimHei']

#数据处理
def get_pretrain_features(file):
    data = pd.read_csv(file)
    y = data[data.columns[0]].values.astype(float)
    level = data[data.columns[1]].values
    test = pd.read_csv('test.csv')
    test = test.values.astype(float)
    l = []
    for i in level:
        l.append(i.replace(" ", ""))
    level = np.array(list(map(float, l))).reshape(-1, 1)
    x = data[data.columns[2:]].values
    x = np.hstack((level, x))
    return x, y, test


def Ensemble(train_features, train_labels, test_features):
    train_features, X_val, train_labels, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

    # 使用早停机制，在验证集的评估指标不再改进时停止训练
    eval_set = [(train_features, train_labels), (X_val, y_val)]

    # 使用xgboost方法进行回归任务
    xgb = XGBRegressor(random_state=42, n_jobs=-1, learning_rate=0.02, n_estimators=500, max_depth=4,min_child_weight= 9
                       , subsample=0.6,colsample_bytree=1,gamma=0.5,reg_alpha=0.2,reg_lambda=0.4)
    clf = xgb

    # 训练模型
    clf.fit(train_features, train_labels, eval_metric="rmse", eval_set=eval_set, early_stopping_rounds=20)
    # joblib.dump(clf, 'model_xgbbest.pkl')

    # 加载模型
    # clf = joblib.load('model_xgb.pkl')

    # 使用KFold
    kf0 = KFold(n_splits=2, random_state=42, shuffle=True)
    kf1 = KFold(n_splits=5, random_state=42, shuffle=True)
    kf2 = KFold(n_splits=10, random_state=42, shuffle=True)
    # 进行交叉验证并计算MSE损失
    scores0 = -1 * cross_val_score(clf, train_features, train_labels, cv=kf0, scoring='neg_mean_squared_error',error_score='raise')
    scores1 = -1 * cross_val_score(clf, train_features, train_labels, cv=kf1, scoring='neg_mean_squared_error',
                                   error_score='raise')
    scores2 = -1 * cross_val_score(clf, train_features, train_labels, cv=kf2, scoring='neg_mean_squared_error',
                                   error_score='raise')

    preds = clf.predict(test_features)
    preds = [int(math.pow(50, i)) for i in preds]
    # 训练数据输入模型后的结果
    preds_train = clf.predict(train_features)


    # 误差下降曲线
    plot_error_history(train_features, train_labels, clf)
    plot_error_estimator(train_features, train_labels, test_features)
    # 绘制特征重要性
    plot_features_importance(train_features, train_labels, clf)
    # 画图对比原数据和预测后数据，看是否拟合
    plot_pred(train_labels, preds_train)

    print("2折交叉验证平均MSE损失：", scores0.mean())
    print("5折交叉验证平均MSE损失：", scores1.mean())
    print("10折交叉验证平均MSE损失：", scores2.mean())
    print("决定系数R^2：", clf.score(train_features, train_labels))
    print("测试输出：", preds)


if __name__ == '__main__':
    file = "data.csv"
    x, y, test = get_pretrain_features(file)
    Ensemble(x, y, test)

