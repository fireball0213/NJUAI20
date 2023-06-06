"""
使用定义好的MLP来预测波士顿房价（506个样本和13个特征）
固定num_epochs=5, batch_size=8：
1.探究数据归一化的影响
2.探究不同的loss function
3.探究不同的learning rate，以及learning rate scheduler的影响
"""

# -*- coding: UTF-8 -*- #
"""
@author:201300086
@time:2023-06-06
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # First hidden layer
        self.h1 = nn.Linear(in_features=13, out_features=20, bias=True)
        self.a1 = nn.ReLU()
        # Second hidden layer
        self.h2 = nn.Linear(in_features=20, out_features=10)
        self.a2 = nn.ReLU()
        # regression predict layer
        self.regression = nn.Linear(in_features=10, out_features=1, bias=False)

    def forward(self, x):
        x = self.h1(x)
        x = self.a1(x)
        x = self.h2(x)
        x = self.a2(x)
        output = self.regression(x)
        return output


def plot_loss(train_losses, criterion, learning_rate):
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('loss_func=' + str(criterion) + ',lr=' + str(learning_rate))
    plt.legend()
    plt.show()


def plot_pred(y_val_tensor, val_outputs, criterion, learning_rate):
    plt.plot(y_val_tensor[:, 0], label='True Values')
    plt.plot(val_outputs.flatten(), label='Predictions')
    plt.xlabel('True Values')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Values with ' + 'loss_func=' + str(criterion) + ',lr_init=' + str(learning_rate))
    plt.show()


def train(model, X_train, y_train, X_val, y_val,
          normalize_flag, fix_lr, num_epochs=5, batch_size=8):
    # Normalize
    if normalize_flag == True:
        scaler = StandardScaler()
        X_train_normalized, X_val_normalized = scaler.fit_transform(X_train), scaler.transform(X_val)
        # 竟然还要再转一次tensor
        X_train, X_val = map(torch.Tensor, (X_train_normalized, X_val_normalized))

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.HuberLoss()

    # learning_rate = 0.001
    # learning_rate = 0.0001
    learning_rate = 0.01

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 每个epoch后学习率衰减gamma倍
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    train_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            labels = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if fix_lr == False:
            scheduler.step()

        train_losses.append(running_loss / len(X_train))

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Training Loss: {running_loss / len(X_train_tensor)},"
              f" Validation Loss: {val_loss.item()},"
              f" Learning Rate:{scheduler.get_lr()}")

    model.eval()

    # Draw the changes of the training loss
    plot_loss(train_losses, criterion, learning_rate)
    # 对比验证集上的预测值和真实值
    plot_pred(y_val_tensor, val_outputs, criterion, learning_rate)

    return model


if __name__ == "__main__":
    boston = load_boston()
    X_train, X_val, y_train, y_val = train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)
    y_train, y_val = y_train.reshape(-1, 1), y_val.reshape(-1, 1)

    # Convert to tensors
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = map(torch.Tensor, (X_train, X_val, y_train, y_val))

    model = MLP()
    model = train(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, normalize_flag=True, fix_lr=False)
