



header 1 | header 2 |header3
---|---|---
row 1 col 1 | row 1 col 2
row 2 col 1 | row 2 col 2



```math
\begin{pmatrix}
0&\infty&\infty&\infty&-1&\infty\\
1&0&\infty&2&\infty&\infty\\
\infty&2&0&\infty&\infty&-8\\
-3&\infty&\infty&0&3&\infty\\
\infty&7&\infty&\infty&0&\infty\\
\infty&5&12&\infty&\infty&0
\end{pmatrix}
```
```math
\begin{pmatrix}
0&6&\infty&\infty&-1&\infty\\
-1&0&\infty&2&0&\infty\\
3&-3&0&4&\infty&-8\\
-3&10&\infty&0&-4&\infty\\
8&7&\infty&9&0&\infty\\
6&5&12&7&\infty&0
\end{pmatrix}
```
```math
\begin{pmatrix}
0&6&\infty&8&-1&\infty\\
-1&0&\infty&2&-2&\infty\\
-2&-3&0&-1&2&-8\\
-3&3&\infty&0&-4&\infty\\
6&7&\infty&9&0&\infty\\
4&5&12&7&10&0
\end{pmatrix}
```
```math
\begin{pmatrix}
0&6&\infty&8&-1&\infty\\
-1&0&\infty&2&-2&\infty\\
-4&-3&0&-1&-3&-8\\
-3&3&\infty&0&-4&\infty\\
6&7&\infty&9&0&\infty\\
4&5&12&7&3&0
\end{pmatrix}
```
```math
\begin{pmatrix}
0&6&\infty&8&-1&\infty\\
-1&0&\infty&2&-2&\infty\\
-4&-3&0&-1&-5&-8\\
-3&3&\infty&0&-4&\infty\\
6&7&\infty&9&0&\infty\\
4&5&12&7&3&0
\end{pmatrix}
```




```math
首先，\frac{\sum_{i}p_i}{\sum_{i,j}c_{ij}}\leq r^*\\
即r^*\sum_{i,j}c_{ij}\geq \sum_{i}p_i\\
若存在负圈，则\sum_{i}w_i\leq 0,即r\sum_{i,j}c_{ij}-\sum_{i}p_i\leq 0\\
于是有r^*\sum_{i,j}c_{ij}\geq \sum_{i}p_i\geq r\sum_{i,j}c_{ij}\\
所以r\leq r^*
```



<p align="left">诶嘿</p>


```math
题意为任意圈满足\sum_{i}w_i\geq 0\\即r\sum_{i,j}c_{ij}-\sum_{i}p_i\geq 0
，\frac{\sum_{i}p_i}{\sum_{i,j}c_{ij}}\leq r\\
而\frac{\sum_{i}p_i}{\sum_{i,j}c_{ij}}的最大值是r^*\\
所以r> r^*
```

