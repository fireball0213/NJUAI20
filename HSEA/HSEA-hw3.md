# HSEA-hw3

###### 201300086史浩男



![image-20221224110157229](C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20221224110157229.png)

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20221224110226792.png" alt="image-20221224110226792" style="zoom:50%;" />

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20221224110235311.png" alt="image-20221224110235311" style="zoom:50%;" />

##### 第一步

**适应层分析法**，先划分解空间：

根据从左开始连续1的个数，划分为n+1个子空间，其中$S_i=\{s\in\{0,1\}^n|f(s)=i\}$



##### 第二步

计算从较低层$S_j$ `jump`到较高层$S_{i}$的概率：

保持从左开始连续的1不变，翻转遇到的第一个0
$$
P(ξ_{t+1}\in\cup_{j=i+1}^{m}S_j\ |\ ξ_t\in S_i)\ge\frac{1}{n}(1-\frac{1}{n})^i
$$

##### 第三步

运行时间公式：
$$
\begin{aligned}
& \sum_{i=0}^{n-1} \pi_{0}\left(S_{i}\right) \sum_{j=i}^{n-1} \frac{1}{v_{j}} \leq \sum_{j=0}^{n-1} \frac{1}{v_{j}} \\
= & \sum_{j=0}^{n-1} n \frac{1}{\left(1-\frac{1}{n}\right)^{i}} \leq \sum_{j=0}^{n-1} n \frac{1}{\left(1-\frac{1}{n}\right)^{n-1}} \leq e n^{2} \in O\left(n^{2}\right)
\end{aligned}
$$


​	



![image-20221224110203744](C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20221224110203744.png)

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20221224110242322.png" alt="image-20221224110242322" style="zoom:67%;" />



### (1)乘性漂移分析

##### 第一步：设计距离函数

$V(x)=n-f(x)$，其中$f(x)$表示x中总共有多少位是1

##### 第二步：计算期望单位漂移距离的下界

在现有$n-x$个0中，翻转一个，其他位不变
$$
E\left[V\left(\xi_{t}\right)-V\left(\xi_{t+1}\right) \mid \xi_{t}=x\right] \geq(n-x) \frac{1}{n}\left(1-\frac{1}{n}\right)^{n-1}=\delta V(x)
$$

##### 第三步：计算期望运行时间上界

$$
\sum_{x \in \chi} \pi_{0}(x) \frac{1+\ln \left(V(x) / V_{m i n}\right)}{\delta}=\sum_{x \in \chi} \pi_{0}(x) \frac{1+\ln (n-f(x))}{\frac{1}{n}\left(1-\frac{1}{n}\right)^{n-1}} \le \frac{1+\ln n}{\frac{e}{n}} \in O(n \log n)
$$





### (2)加性漂移分析

##### 第一步：设计距离函数

$V(x)=n-f(x)$，其中$f(x)$表示x中总共有多少位是1

##### 第二步：计算期望单位漂移距离的下界

在现有$n-i$个0中，翻转一个，其他位不变
$$
E\left[V\left(\xi_{t}\right)-V\left(\xi_{t+1}\right) \mid \xi_{t}=x\right] \geq(n-i) \frac{1}{n}\left(1-\frac{1}{n}\right)^{n-1}=c_l
$$

##### 第三步：计算期望运行时间上界

$$
\sum_{x \in \chi} \pi_{0}(x) \frac{V(x) }{c_l}\leq\frac{n }{c_l}=en^2 \in O(n^2)
$$





![image-20221224110209178](C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20221224110209178.png)

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20221224110255920.png" alt="image-20221224110255920" style="zoom:67%;" />

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20221224110305709.png" alt="image-20221224110305709" style="zoom:67%;" />

#### 1、问题分析：

分析COCZ问题的解，发现所有帕累托最优解恰好是**所有前$n/2$位**都是1的解

观察GSEMO算法，发现只有可比的解之间才会发生替换，也就是说随着算法的运行，解空间中解的个数**严格递增**；且已达到帕累托最优解的个体不会被替换掉，所以解空间中帕累托最优解的个数也是**严格递增**、



#### 2、解决思路：

因此可以把解决问题拆成两部分：

1. 算法找到互相之间不可比的n/2个解组成的解空间，即种群P中解的个数达到最大值。

   这一部分，可以看成同时处理n/2个类似`OneMax`的问题，其中每个类似`OneMax`的问题都在优化n位的01串的**后n/2位**，优化目标为后n/2位中共有i个1。

   

   

2. 种群P中解的个数不变，只对每个解进行可能的优化，直到所有解都成为帕累托最优解。

   这一部分，可以看成同时处理n/2个`OneMax`问题，其中每个`OneMax`问题都在优化n位的01串的**前n/2位**，优化目标为前n/2位全是1



​		

#### 3、引理--OneMax的推广：

定义一类与和`OneMax`相似的问题，我称之为`类OneMax`问题。`类OneMax`与`OneMax`的唯一区别在于，`OneMax`的优化目标是`111...111`，而`类OneMax`的优化目标是同样长的任意01串

下面证明，`类OneMax`的解决的时间上界都和`OneMax`相同，是$O(n \log n)$



##### 		第一步，划分解空间：

​		根据从左开始连续满足优化目标的位数个数，划分为n+1个子空间，其中$S_i=\{s\in\{0,1\}^n|s中的前i位满足优化目标\}$



##### 		第二步：

​		计算从较低层$S_j$ `jump`到较高层$S_{i}$的概率：

​		保持其他位数不变，翻转从左开始遇到的第一个不满足优化目标的位
$$
P(ξ_{t+1}\in\cup_{j=i+1}^{m}S_j\ |\ ξ_t\in S_i)\ge\frac{n-i}{n}(1-\frac{1}{n})^{n-1}
$$

##### 		第三步

​		运行时间公式：
$$
\begin{aligned}
& \sum_{i=0}^{n-1} \pi_{0}\left(S_{i}\right) \sum_{j=i}^{n-1} \frac{1}{v_{j}} \leq \sum_{j=0}^{n-1} \frac{1}{v_{j}} \\
\le & \sum_{j=0}^{n-1}\frac{n}{n-j} \frac{1}{\left(1-\frac{1}{n}\right)^{n-1}}  \leq e n\sum_{j=1}^{n}\frac{1}{j} \in O(n \log n)
\end{aligned}
$$


#### 4、第一部分解决方法：

把“找到互相之间不可比的n/2个解”分解成“找到特定的n/2个解”：

就像`OneMax`问题的目标是找到`111...111`一样，我把问题分解成找到最后n/2位分别为`000..000`,`1000...000`,`1100...000`,`1110...000`,......,`111...100`,`111...110`,`111...111`的这n/2个`类OneMax`子问题，这些子问题代表了所有互相之间不可比的解。由于解空间中解的个数**严格递增**，因此解出的每个子问题都不会被算法后续步骤破坏掉。解出这些子问题，也就找到了互相之间不可比的n/2个解。



根据引理，可以在$n*O(n \log n)=O(n^2 \log n)$时间内解决这n个子问题，也就解决了第一部分，得到了包含互相之间不可比的n/2个解组成的解空间



#### 5、第二部分解决方法：

此时已得到互相之间不可比的n/2个解，只需对每个解进行优化，直到所有解都成为帕累托最优解。

其中每个解的优化都是优化目标为前n/2全为1的`类OneMax`问题，一共有n/2个`类OneMax`问题，一共需要$n*O(n \log n)=O(n^2 \log n)$的时间上界



#### 6、结论

因此，总时间上界为$O(n^2 \log n)+O(n^2 \log n)=O(n^2 \log n)$