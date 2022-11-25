

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20220928155951035.png" alt="image-20220928155951035" style="zoom: 50%;" />



<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20220928160009375.png" alt="image-20220928160009375" style="zoom: 67%;" />

#### (1)

$$
W^{[2]}\in R^{K\times D_a}\ \ b^{[2]}\in R^{K\times 1}\\
输出维度K\times m
$$

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20220928162001974.png" alt="image-20220928162001974" style="zoom: 67%;" />

#### (2)

$$
\begin{aligned}
\frac{\partial \hat{y}_{k}}{\partial z_{k}^{[2]}}
&=\frac{\partial(\frac{e^ {z_{k}^{[2]}}}{\sum_{j=1}^{K} e^{\boldsymbol{z_{j}^{[2]}}}})}{\partial z_{k}^{[2]}}\\
&=\frac{\partial(1-\sum_{j\not=k} e^{\boldsymbol{z_{j}^{[2]}}}\times\frac{1}{\sum_{j=1}^{K} e^{\boldsymbol{z_{j}^{[2]}}}})}{\partial z_{k}^{[2]}}\\
&=-\sum_{j\not=k} e^{\boldsymbol{z_{j}^{[2]}}}\frac{\partial(\frac{1}{\sum_{j=1}^{K} e^{\boldsymbol{z_{j}^{[2]}}}})}{\partial z_{k}^{[2]}}\\
&=\sum_{j\not=k} e^{\boldsymbol{z_{j}^{[2]}}}{\frac{e^{\boldsymbol{z_{k}^{[2]}}}}{Z^2}}\\
&=\frac{(Z-e^{\boldsymbol{z_{k}^{[2]}}})e^{\boldsymbol{z_{k}^{[2]}}}}{Z^2}\\
&=(1-\hat{y}_{k})\hat{y}_{k}
\end{aligned}
$$

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20220928163223167.png" alt="image-20220928163223167" style="zoom:67%;" />

#### (3)

$$
\begin{aligned}
\frac{\partial \hat{y}_{k}}{\partial z_{i}^{[2]}}
&=\frac{\partial(\frac{e^ {z_{k}^{[2]}}}{\sum_{j=1}^{K} e^{\boldsymbol{z_{j}^{[2]}}}})}{\partial z_{i}^{[2]}}\\
&= e^{\boldsymbol{z_{k}^{[2]}}}\frac{\partial(\frac{1}{\sum_{j=1}^{K} e^{\boldsymbol{z_{j}^{[2]}}}})}{\partial z_{i}^{[2]}}\\
&= -e^{\boldsymbol{z_{k}^{[2]}}}{\frac{e^{\boldsymbol{z_{i}^{[2]}}}}{Z^2}}\\
&=-\hat{y}_{i}\hat{y}_{k}
\end{aligned}
$$

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20220928163742270.png" alt="image-20220928163742270" style="zoom:67%;" />

#### (4)

$$
\begin{aligned}
\frac{\partial L}{\partial z_{i}^{[2]}}
&=-\frac{\partial(y_k\log(\hat y_k))}{\partial z_{i}^{[2]}}\\
&=-\frac{\partial\log(\hat y_k)}{\partial z_{i}^{[2]}}\\
\end{aligned}
$$

若$k=i:$
$$
\begin{aligned}
\frac{\partial L}{\partial z_{i}^{[2]}}
&=-\frac{\partial\log(\hat y_i)}{\partial z_{i}^{[2]}}\\
&=-\frac{\partial(z_{i}^{[2]}-\log(Z))}{\partial z_{i}^{[2]}}\\
&=\hat y_i-1
\end{aligned}
$$
若$k\not=i:$
$$
\begin{aligned}
\frac{\partial L}{\partial z_{}^{[2]}}
&=-\frac{\partial\log(\hat y_k)}{\partial z_{i}^{[2]}}\\
&=-\frac{\partial(z_{k}^{[2]}-\log(Z))}{\partial z_{i}^{[2]}}\\
&=\hat y_i
\end{aligned}
$$
