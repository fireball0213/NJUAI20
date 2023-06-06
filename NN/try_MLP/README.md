# 使用MLP探究神经网络优化

探究数据归一化、固定/递减学习率、不同参数初始化方法、不同的参数归一化方法对性能影响  



## 任务一：`MLP_boston.py`

使用定义好的**MLP**来预测波士顿房价（**506**个样本和**13**个特征）使用**SGD**优化器，固定**num_epochs=5, batch_size=8**：

1.探究数据归一化的影响

2.探究不同的**loss function的影响**

3.探究不同的固定**learning rate**，以及**learning rate scheduler**的影响



> 使用SGD而不用Adam，是为了放大变量的一影响
>
> 详细过程见实验报告



### 结论：

- 数据必须归一化，否则网络跑不动，也没意义
- 在本问题的固定参数设置上，MSE是合理的Loss函数
- 基于以上设置，合理的学习率只有0.001这个量级最合适，使用学习率衰减时也要接近这个量级才有预测效果


  
## 任务二：`MLP_MINIST.py`

使用**MNIST**数据集，训练一个**MLP**模型，包含两个全连接层，第一层包含**2048**个神经元，第二层包含**10**个神经元，激活函数为**ReLU**。
使用**SGD**优化器，学习率**0.001**，**5**个**epoch**，**batch size**为**100**，损失函数为交叉熵损失

1.探究不同参数初始化方法（**Default,Uniform,Normal,Xavier,Kaiming**）对模型训练的影响。

2.探究不同的归一化方法（BN、LN、IN、GN)对模型训练的影响。

3.探究归一化方法和参数初始化方法有哪些复杂的相互作用



> 使用SGD而不用Adam，是为了放大变量的一影响
>
> 详细过程见实验报告



### 结论：

#### 1、不使用归一化方法时：（对比5种参数初始化方式）

- 默认初始化的效果最差，且验证集性能迟迟得不到提升
- 其余4种方法中，高斯分布初始化收敛慢，因此在仅有的5个epoch内效果最差。但扩大到20epoch时仍是最差的
- He方法适用于Relu，果然优于Xavier

#### 2、使用归一化方法时：（在MLP中添加归一化层，对比5种参数初始化方式在4种归一化方法上的区别）

- 对于默认初始化参数的情况，使用BN后性能发生翻天覆地的变化，超过了其他所有初始化方法+BN的性能
- 但Uniform初始化参数的方法在使用BN后，性能几乎没有变化
- ==在使用BN前越差的初始化方法，使用BN后性能越好，性能顺序完全逆转！！！==

##### 原因分析：

- 不使用 BN 时，默认初始化的权重可能存在较大的变化范围，导致网络训练时的不稳定，所以效果差，初始化越合理的方法效果越好
- 使用BN时， BN 可以稳定每层的输入分布、稳定每层的激活输出， 所以较差的初始化方法带来的问题可以被大大减轻，降低了初始化权重的影响。反而较差的初始化方法的权重范围更大，提供了更多的可能性，使得网络有机会找到更好的解，**所以越野蛮的参数初始化反而在BN的加持下效果越好**。
- 对于 Uniform 初始化来说，在不使用 BN 时就已经限制了权重的变化范围，使得训练稳定，因此使用 BN 后，可能不会像默认初始化那样带来显著的提升。



#### 3、归一化方法和参数初始化方法可能会有一些复杂的相互作用：

- 在使用归一化方法前越差的参数初始化方法，归一化后性能越好，性能顺序完全逆转！！！
- GN归一化方法提升最少，甚至在Default和Uniform初始化方法中出现了性能倒退

![img](./README.assets/LQRC}S6`JZ5P[J480$G96KS.gif)

##### 打个比方理解这件事：

不同的初始化方法，就像不同地方来的基础不同的小兵，有能打的，有不能打的。

归一化方法，就像是统一穿上铠甲，进行操练。

结果操练过后，原来最不能打的，变成最能打。原来最能打的，几乎没啥进步。

放到职场里，这些原来不能打的小兵，可能就是应届生，调教之后能比有工作经历的还能打。

**所以这或许解释了，为什么应届生身份很重要...**

在归一化方法很强大的时候，初始化越野蛮越好

### 
