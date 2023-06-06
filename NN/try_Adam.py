
"""
使用Adam梯度下降法求解Himmelblau函数的最小值
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def him(x):
    return (x[0] ** 2+x[1]-11) ** 2+(x[0]+x[1] ** 2-7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
Z = him([X, Y])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='rainbow')
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.set_zlabel('f')
plt.show()


x = torch.tensor([0., 0.], requires_grad=True)#初始化需要优化的参数
optimizer = torch.optim.Adam([x], lr=1e-2) # 学习率

for step in range(1000):
    pred = him(x)
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()
    if (step+1) % 100 == 0:#每100epoch输出
        print(f"step={step+1}, x={x.tolist()}, f(x)={pred.item()}")
