# 作业: 自编码器

> Deadline: 12-02 23:59:59 
> 
> 提交网址: [深度学习平台作业提交 (nju.edu.cn)](https://table.nju.edu.cn/dtable/forms/0572926f-1534-4b21-aff7-833435d4a885/)

本次作业我们要实现自编码器, 包括AutoEncoder, Variational AutoEncoder, Conditional Variational AutoEncoder.

前置知识:

- Python语言基础, 这样有助于你写出更加高效的代码
- Pytorch 基础, 会写基础的模型与训练代码
- 理论基础, 掌握最大似然估计, 贝叶斯推断, ELBO 等理论推导

当然, 如果这些你都有些欠缺, 也不要担心, 我们可以参考网络上很好的教程, 下面是几个非常不错的参考资料:

- [CVAE, pyro的文档讲解(英文), 有代码](https://pyro.ai/examples/cvae.html)
- [CVAE, TensorFlow的文档(中文), 有代码](https://www.tensorflow.org/tutorials/generative/cvae?hl=zh-cn)
- [VAE, 原理讲解(英文), 可能需要科学上网](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

注意! 请不要分享你的代码给同学或是上传到网络上, 这样可能会让其他同学失去学习的乐趣, 也有害于学术诚信. 
本次作业难度不高, 愿意花一点时间的同学都可以在几天的时间里完成并且获得进步! Enjoy it!
遇到问题的操作顺序:

- 自己动手debug, 思考为什么会有问题
- 搜索报错的问题, 或是相关的内容
- 与同学们讨论, 但是不能抄袭
- 群里@助教, 不过在条件允许的情况下, 请先独立尝试.

你的作业应该包括下面的内容:

- 能够运行的代码, 并且具有良好的注释, 方便助教给分
- 能够体现你工作量的报告, 包括 简单的说明任务, 代码的超参数设置, 不同的自编码器之间的对比(包括理论上与实验上), 自己的思考

## 三种自编码器之间的区别

AutoEncoder 提出来主要是为了解决 **数据压缩与还原**的问题, 我们输入一个 x, 通过encoder得到隐变量 z, 再将 z 经过 decoder 的处理还原出 x'. 于是很自然的, 优化目标就应该是希望 \textbf{dist}(x,x') 尽量小.

Variational AutoEncoder 提出来是为了做 **生成式的任务**, 希望模型可以通过学习, 获得如何生成和输入的 x 很像但是不完全一样的x', 比如我们希望生成动漫头像, 我们当然是希望生成的头像与训练的数据是风格相似但是不完全一样的. 为了做到这一点, 我们采取变分推断的方式, 不再是通过encoder获得一个隐变量, 而是生成这个隐变量服从的分布的均值与方差(实际上这里我们假设隐变量服从高斯分布, 那么均值和方差实际上就唯一刻画了这个分布的所有信息). 再经过decoder的处理得到 x', 此时的优化目标除了让 \textbf{dist}(x,x') 尽可能小以外, 还应该满足对隐变量获得的分布的约束 \textbf{KL}(P_z, prior_z) , 也就是我们希望我们的z服从的分布与我们给定的先验分布(这里是高斯分布)也尽量接近.

Conditional Variational AutoEncoder 则是在VAE的基础上, 加入了对数据类别信息的指定, 从而我们可以做到指定模型生成具体某个类别数据.

## 作业说明

本次作业我们的代码结构是这样的:

```
- .
 |
 |- data.py # 对数据集的处理, 包括得到dataloader的功能
 |- main.py # 主文件, 包括对train过程的实现与损失函数的实现 (分数 45/100)
 |- model.py # 模型实现, 包括对AE, VAE, CVAE的实现 (分数 55/100)
```

你需要补全或是重构我们提供的这份代码文件, 按照我们的注释补全相应内容或是自己重新另起炉灶都是可以的, 但是请在对应的地方写清注释, 方便我们给分.

举个例子, 在`main.py`中:

```python
    def kl_div(self, p, q):
        # 实现kl散度的损失 (5/100)
        return
```

你需要做的是在这个提示的地方, 补全相应的功能, 并且能让你的代码最终运行起来. 我们提供的框架只是一种可能的风格, 请放心随意修改.

在报告中, 你应该展示出三种自编码器在MNIST数据集上的生成结果:

- 对于AE, 你可以考虑展示出压缩前后图片的不同
  
  ![](https://blog.keras.io/img/ae/sparse_ae_32.png)

- 对于VAE, 你可以从\mathcal{N}(0,1)中采样一些点作为隐变量, 生成一些图片查看结果
  
  <img src="https://img2020.cnblogs.com/blog/2226924/202104/2226924-20210423185609828-1496768811.jpg" title="" alt="" width="299">

- 对于CVAE, 你可以指派具体的标签, 从\mathcal{N}(0,1) 中采样, 生成一些图片查看结果 ![](https://pic1.imgdb.cn/item/636da3bb16f2c2beb14394d8.png)

> 以上的展示方式并不唯一, 你可以设计你认为适合的展示方式来体现出不同自编码器的效果.

## 可能的完成路线

### Step1 (`model.py`)

首先我们需要实现 `Encoder` 和 `Decoder` 这两个类,

```python
class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size, is_dist=False, **kwargs) -> None:
        super(Encoder, self).__init__()
        self.mu = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size),)
        if is_dist:  # 如果需要encoder返回的是均值与标准差, 那么我们需要额外引入一次计算标准差的layer
            self.sigma = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size),)

    def forward(self, xs):
        # 实现编码器的forward过程 (5/100), 注意 is_dist 的不同取值意味着我们需要不同输出的encoder
        ...
        return
```

这里, 我们为AE,VAE,CVAE留了一个接口, 就是借助`is_dist`这个参数来判断我们的`encoder`是输出一个隐变量(AE)呢, 还是输出隐变量服从的分布的均值与方差(VAE,CVAE)呢.

```python
class Decoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size, decode_type="AE", **kwargs) -> None:
        super(Decoder, self).__init__()
        if decode_type == "AE":
            self.decoder = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, x_dim),)
        elif decode_type == "VAE":
            self.decoder = ...  # 实现VAE的decoder (5/100)
        elif decode_type == "CVAE":
            self.decoder = ...  # 实现CVAE的decoder (5/100)
        else:
            raise NotImplementedError

    def forward(self, zs, **otherinputs):
        # 实现decoder的decode部分, 注意对不同的decode_type的处理与对**otherinputs的解析 (10/100)
        return ...
```

Decoder的实现稍微复杂一些, 考虑到不同的自编码器可能需要不同的输入, 我们这里统一借助`**otherinputs`来处理.
对于AE的情况, 那么我们显然只需要`zs`作为输入即可, 对于VAE,CVAE的情况, 我们可能需要隐变量服从分布的均值与方差(对于CVAE, 还需要类别的指示变量)

在实现遇到迷茫的时候, 不妨考虑具体的自编码器类需要什么样的encoder, decoder.

```python
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        # 实现AE的forward过程(5/100)
        ...
        return


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs):
        mu, sigma = self.encoder(xs)
        # 实现VAE的forward过程(10/100)
        ...
        return


class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(ConditionalVariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs, ys):
        mu, sigma = self.encoder(xs, ys)
        # 实现 CVAE的forward过程(15/100)
        ...
        return
```

当我们将所有的模型都实现好了, 你可以写一些简单的测试用例来验证输入输出的维度是否对齐, torch的debug不过如此... 一杯茶一根烟, 一行代码debug一天...

### Step2 (`main.py`)

模型实现好了, 那么我们来训练吧!
这里就简单了, 只需要实现训练的loss, 注意一下不同自编码器可能需要不同的loss就可以了, 当然你也可以把超参数诸如各个不同loss部分之间的比例也当作args传进去...

至于训练过程, 不过就是optimizer.step()搞一搞就行了 :)
