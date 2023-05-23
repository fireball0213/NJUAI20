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
    death = data[data.columns[2]].values
    hurt = data[data.columns[3]].values
    depth = data[data.columns[4]].values
    test = pd.read_csv('test.csv')
    test = test.values.astype(float)
    l = []
    for i in level:
        l.append(i.replace(" ", ""))
    level = np.array(list(map(float, l))).reshape(-1, 1)
    x = data[data.columns[2:]].values
    x = np.hstack((level, x))
    return x, y, test


def plot_error(train_features, train_labels, test_features):

    n_estimators = [50, 100, 150, 200, 250, 300]
    train_errors = []

    for n in n_estimators:
        xgb = XGBRegressor(random_state=42, n_jobs=-1, learning_rate=0.1, n_estimators=n, max_depth=5)
        xgb.fit(train_features, train_labels)
        train_pred = xgb.predict(train_features)
        train_errors.append(mean_squared_error(train_labels, train_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(n_estimators, train_errors, label='Training error')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Number of Estimators')
    plt.title('XGBoost - Effect of n_estimators on error')
    plt.legend()
    plt.show()


def plot_error_history(train_features, train_labels,clf):
    eval_set = [(train_features, train_labels)]
    clf.fit(train_features, train_labels, eval_metric="rmse", eval_set=eval_set, verbose=True)

    # 获取历史误差
    results = clf.evals_result()
    epochs = len(results['validation_0']['rmse'])

    # 绘制历史误差
    fig, ax = plt.subplots()
    ax.plot(range(0, epochs), results['validation_0']['rmse'], label='Train')
    ax.legend()
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.title('XGBoost MSE')
    plt.show()

def plot_features_importance(train_features, train_labels,clf):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title("Feature importances")
    plt.bar(range(train_features.shape[1]), importances[indices],
            color="purple", align="center")
    feature_names = ["level", "death", "hurt", "depth"]  # 特征的名称
    plt.xticks(range(train_features.shape[1]), [feature_names[i] for i in indices], rotation=0, c="red")
    plt.xlim([-1, train_features.shape[1]])
    plt.show()

#计算均方误差的函数
def rmse(y_test, y):
    return np.sqrt(mean_squared_error(y_test, y))

#计算MSE的函数
def mse(y_test, y):
    return mean_squared_error(y_test, y)

def Ensemble(train_features, train_labels, test_features):
    train_features, X_val, train_labels, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)
    # 使用线性回归基学习器进行回归任务
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)  # , oob_score=True
    bag = BaggingRegressor(random_state=42, base_estimator=rf, n_estimators=10, n_jobs=-1)
    gb = GradientBoostingRegressor(random_state=42)  # 0.783
    # Stacking学习法集成
    # clf = StackingRegressor(estimators=[('rf', rf), ('bag', bag)],final_estimator=xgb, passthrough=True, verbose=1,n_jobs=-1)
    # Voting投票法集成
    # clf = VotingRegressor(estimators=[('rf', rf),  ('xgb', xgb),  ('gb', gb)])

    # 使用早停机制，在验证集的评估指标不再改进时停止训练
    eval_set = [(train_features, train_labels), (X_val, y_val)]

    # 使用xgboost方法进行回归任务
    xgb = XGBRegressor(random_state=42, n_jobs=-1, learning_rate=0.02, n_estimators=500, max_depth=4,min_child_weight= 9
                       , subsample=0.6,colsample_bytree=1,gamma=0.5,reg_alpha=0.2,reg_lambda=0.4)
    #使用网格搜索
    #'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1], 'gamma': [0, 0.1, 0.2, 0.3, 0.4,0.5],
    #    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],'learning_rate': [0.02, 0.05,0.08, 0.1],
    #    'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'subsample': [0.6, 0.7, 0.8, 0.9, 1]
    #
    # param_grid = {'reg_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5], 'reg_lambda': [0, 0.1, 0.2, 0.3, 0.4, 0.5] }
    # grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error')
    # grid_search.fit(train_features, train_labels, eval_metric="rmse", eval_set=eval_set, early_stopping_rounds=20, verbose=True)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)

    ada = AdaBoostRegressor(base_estimator=rf,learning_rate=0.01, n_estimators=100, random_state=42)
    # 使用网格搜索调参ada
    # param_grid = {'learning_rate': [ 0.001, 0.01, 0.1,0.02,0.005], 'n_estimators': [50, 100, 200, 300, 400, 500],}
    # grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='neg_mean_squared_error')
    # grid_search.fit(train_features, train_labels)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)

    clf = xgb
    # clf=ada


    # 训练模型
    clf.fit(train_features, train_labels, eval_metric="rmse", eval_set=eval_set, early_stopping_rounds=20)
    # clf.fit(train_features, train_labels)
    joblib.dump(clf, 'model_xgbbest.pkl')

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


    print("交叉验证平均MSE损失：", scores0.mean(),scores1.mean(),scores2.mean())
    print("决定系数R^2：", clf.score(train_features, train_labels))
    # print("全部数据MSE，RMSE：",mse(train_labels,preds_train),rmse(train_labels,preds_train))
    print("测试输出：", preds)

    # 误差下降曲线
    # plot_error_history(train_features, train_labels, clf)
    # 绘制特征重要性
    # plot_features_importance(train_features, train_labels, clf)
    # 画图对比原数据和预测后数据，看是否拟合
    # plt.plot(train_labels, color='blue', label='true', linestyle='-', marker='o')
    # plt.plot(preds_train, color='red', label='predict', linestyle='-', marker='+')
    # # 调整图纸大小.使宽度更大
    # plt.rcParams['figure.figsize'] = (20.0, 8.0)  # 单位是inches
    # # 给图像添加横纵坐标
    # plt.title('true and predict')
    # plt.xlabel('data')
    # plt.ylabel('damage')
    # plt.legend()
    # plt.show()





if __name__ == '__main__':
    file = "data.csv"
    x, y, test = get_pretrain_features(file)
    Ensemble(x, y, test)
    # plot_error(x,y, test)
