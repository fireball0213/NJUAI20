from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

X, y = load_boston(return_X_y=True)
trainx, testx, trainy, testy = train_test_split(
    X, y, test_size=0.33, random_state=42)

def linReg(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:# linear regression
    #ones(x)创建x维1向量，shape[0]行数，reshape(-1, 1)变为一列，np.hstack横着拼接
    # transpose转置，@矩阵乘法，np.linalg.inv求逆，T：换元
    X_expand = np.hstack((X_train, np.ones(X_train.shape[0]).reshape(-1, 1)))
    T: np.ndarray = np.linalg.inv(X_expand.transpose() @ X_expand)
    return T @ X_expand.transpose() @ y_train

def linRegMSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    w = linReg(X_train, y_train).reshape(-1, 1)
    X_expand = np.hstack((X_test, np.ones(X_test.shape[0]).reshape(-1, 1)))
    E: np.ndarray = y_test.reshape(-1, 1) -  X_expand @ w
    return np.mean(E ** 2)

def ridgeReg(X_train: np.ndarray, y_train: np.ndarray, lmbd: float) -> np.ndarray:# ridge regression
    #np.eye单位矩阵
    l=np.ones(X_train.shape[0])
    T: np.ndarray = np.linalg.inv(X_train.transpose() @ X_train + 2 * lmbd * np.eye(X_train.shape[1]))
    TT: np.ndarray = l @ X_train @ T @ X_train.transpose() - l
    b = (TT @ y_train) / (TT @ l)
    w = T @ X_train.transpose() @ (y_train - b * l)
    return np.hstack((w, b))


def ridgeRegMSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, lmbd: float) -> float:
    w = ridgeReg(X_train, y_train, lmbd).reshape(-1, 1)
    X_expand = np.hstack((X_test, np.ones(X_test.shape[0]).reshape(-1, 1)))
    E: np.ndarray = y_test.reshape(-1, 1) - X_expand @ w
    return np.mean(E ** 2)

# print(linReg(trainx, trainy))
#print(ridgeReg(trainx, trainy, 0))
print(linRegMSE(trainx, trainy, testx, testy))
print(ridgeRegMSE(trainx, trainy, testx, testy, 0))
lmbds = np.arange(0, 3.1, 0.25)
mse = [ridgeRegMSE(trainx, trainy, testx, testy, lmbd) for lmbd in lmbds]
print(lmbds)
print(mse)
plt.plot(lmbds, mse)
plt.xlabel('λ')
plt.ylabel('ridgeRegMSE')
plt.scatter(lmbds, mse,marker = 'o')
plt.show()