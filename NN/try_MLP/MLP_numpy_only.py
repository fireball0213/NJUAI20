"""
仅使用numpy实现MLP，不使用深度学习框架
探索不同激活函数：sigmoid, tanh, relu, leaky_relu的手动实现
"""

# -*- coding: UTF-8 -*- #
"""
@author:201300086
@time:2023-06-09
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import dataset_loader, timer


class MLP(object):
    def __init__(self, input_size, hidden_size, output_size, active):
        # weight initialization
        self.activation = active
        if self.activation == 'relu' or self.activation == 'leaky_relu':
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        else:
            self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
            self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def tanh(self, z):
        return np.tanh(z)

    def tanh_prime(self, z):
        return 1 - np.tanh(z) ** 2

    def relu(self, z):
        return np.maximum(0, z)

    def relu_prime(self, z):
        return (z > 0).astype(int)

    def leaky_relu(self, z, alpha=0.01):
        return np.maximum(alpha * z, z)

    def leaky_relu_prime(self, z, alpha=0.01):
        dz = np.ones_like(z)
        dz[z < 0] = alpha
        return dz

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        # hidden_layer_input
        self.z1 = x.dot(self.W1) + self.b1
        # hidden_layer_output
        if self.activation == 'sigmoid':
            self.a1 = self.sigmoid(self.z1)
        elif self.activation == 'tanh':
            self.a1 = self.tanh(self.z1)
        elif self.activation == 'relu':
            self.a1 = self.relu(self.z1)
        elif self.activation == 'leaky_relu':
            self.a1 = self.leaky_relu(self.z1)
        # output_layer_input
        self.z2 = self.a1.dot(self.W2) + self.b2
        y_hat = self.softmax(self.z2)
        return y_hat

    def backward(self, x, y, y_hat, learning_rate):
        m = x.shape[0]
        d2 = y_hat
        d2[range(m), y] -= 1
        dW2 = (self.a1.T).dot(d2)
        db2 = np.sum(d2, axis=0, keepdims=True)
        if self.activation == 'sigmoid':
            d1 = d2.dot(self.W2.T) * self.sigmoid_prime(self.z1)
        elif self.activation == 'tanh':
            d1 = d2.dot(self.W2.T) * self.tanh_prime(self.z1)
        elif self.activation == 'relu':
            d1 = d2.dot(self.W2.T) * self.relu_prime(self.z1)
        elif self.activation == 'leaky_relu':
            d1 = d2.dot(self.W2.T) * self.leaky_relu_prime(self.z1)
        dW1 = np.dot(x.T, d1)
        db1 = np.sum(d1, axis=0)

        # update parameters
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, x_train, y_train, batch_size, learning_rate):
        num_samples = x_train.shape[0]
        random_indices = np.random.permutation(num_samples)
        x_train = x_train[random_indices]
        y_train = y_train[random_indices]

        for i in range(0, num_samples, batch_size):
            x_train_batch = x_train[i:i + batch_size]
            y_train_batch = y_train[i:i + batch_size]

            y_hat_batch = self.forward(x_train_batch)
            self.backward(x_train_batch, y_train_batch, y_hat_batch, learning_rate)

    def evaluate(self, x_test, y_test):
        y_hat_test = self.forward(x_test)
        y_pred_test = np.argmax(y_hat_test, axis=1)
        accuracy = np.mean(y_pred_test == y_test)
        # 保留四位小数
        accuracy = round(accuracy, 4)
        return accuracy


@timer
def MLP_numpy_only(epochs, batch_size, learning_rate, active='sigmoid'):
    train_image, train_label, test_image, test_label = dataset_loader()
    model = MLP(784, 30, 10, active=active)

    #计算网络的参数量
    print()
    print("the number of parameters is {}".format(model.W1.size + model.W2.size + model.b1.size + model.b2.size))
    #网络的计算量
    print("the number of flops is {}".format(2 * model.W1.size + 2 * model.W2.size + model.b1.size + model.b2.size))

    train_accuracy_list = []
    test_accuracy_list = []
    #记录测试集上准确率最优的模型准确率，并输出。使用早听机制，当测试集上准确率连续5次没有出现最优后，停止训练
    best_accuracy = 0
    count = 0
    for epoch in range(epochs):
        model.train(train_image, train_label, batch_size, learning_rate)
        train_accuracy = model.evaluate(train_image, train_label)
        test_accuracy = model.evaluate(test_image, test_label)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            count = 0
        else:
            count += 1
        if count == 5:
            break
        print("epoch:{}/{} train/test acc: {}/{} using {}".format(epoch, epochs, train_accuracy, test_accuracy, active))
    print("the best accuracy is {} ----{}".format(best_accuracy, active))

    # 画出准确率随epoch变化的曲线，在图中同时对比训练准确率和测试准确率
    plt.plot(range(len(train_accuracy_list)), train_accuracy_list, label='train')
    plt.plot(range(len(test_accuracy_list)), test_accuracy_list, label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('train vs test using ' + str(active))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    epochs = 50
    batch_size = 128
    learning_rate = 0.01

    MLP_numpy_only(epochs, batch_size, learning_rate, active='sigmoid')
    MLP_numpy_only(epochs, batch_size, learning_rate, active='tanh')
    # MLP_numpy_only(epochs, batch_size, learning_rate, active='relu')
    MLP_numpy_only(epochs, batch_size, learning_rate, active='leaky_relu')
