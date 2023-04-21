# -*- coding: UTF-8 -*- #
"""
@filename:PCA-第五题.py
@author:201300086
@time:2023-04-13
"""
import numpy as np
import matplotlib.pyplot as plt

def pca(x):
    x_c = x - x.mean(axis=0)
    cov_matrix = (1 / x.shape[0]) * (x.T @ x)
    ev, em = np.linalg.eig(cov_matrix)
    idx = np.argsort(ev)[::-1]
    P = em[:, idx[:x.shape[1]]]
    x_pca = x_c @ P
    return x_pca

def pca_whiten(x):
    x_c = x - x.mean(axis=0)
    cx = (1 / x.shape[0]) * (x.T @ x)
    eas, evs = np.linalg.eig(cx)
    idx = np.argsort(eas)[::-1]
    ead = eas[idx[:x.shape[1]]]
    evd = evs[:, idx[:x.shape[1]]]
    ep = 1e-5
    x_w = x_c @ evd / np.sqrt(ead + ep)
    return x_w

#任务一
X = np.random.randn(2000, 2) @ np.array([[2, 1], [1, 2]])
plt.scatter(X.T[0], X.T[1], c='purple', alpha=0.5)
plt.show()

#任务二
X_whiten = pca(X)
plt.scatter(X_whiten.T[0], X_whiten.T[1], c='purple', alpha=0.5)
plt.show()

#任务三
X_whiten = pca_whiten(X)
plt.scatter(X_whiten.T[0], X_whiten.T[1],c='purple', alpha=0.5)
plt.show()