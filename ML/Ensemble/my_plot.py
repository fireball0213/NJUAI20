# -*- coding: UTF-8 -*- #
"""
@filename:my_plot.py
@author:201300086
@time:2023-05-24
"""

import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error

font = {'family': 'Times New Roman', 'size': 18}
matplotlib.rc('font', **font)
matplotlib.rcParams.update(
    {'font.family': 'Times New Roman', 'font.size': 18, })
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.sans-serif'] = ['SimHei']


def plot_error_estimator(train_features, train_labels, test_features):
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


def plot_error_history(train_features, train_labels, clf):
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


def plot_features_importance(train_features, train_labels, clf):
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

def plot_pred(train_labels,preds_train):
    plt.plot(train_labels, color='blue', label='true', linestyle='-', marker='o')
    plt.plot(preds_train, color='red', label='predict', linestyle='-', marker='+')
    # # 调整图纸大小.使宽度更大
    plt.rcParams['figure.figsize'] = (20.0, 8.0)  # 单位是inches
    # # 给图像添加横纵坐标
    plt.title('true and predict')
    plt.xlabel('data')
    plt.ylabel('damage')
    plt.legend()
    plt.show()
