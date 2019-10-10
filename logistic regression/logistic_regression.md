# Logistic Regression

## 概述

Logistic regression 是一种监督学习模型，虽然名字中包含 regression （回归），但实则用于处理分类问题。Logsitic regression 基于 linear regression （线性回归）的思想，将其输出值作为 sigmoid 函数的输入，最后输出值范围为 [0, 1]，以 0.5 为分界点将样本分为两类。另外也有其他的方法使得 logistic regression 支持处理多分类问题。

## 优点

- 模型高效（计算量小，占用内存少），适用于大数据场景
- 可解释性好

## 缺点

- 不原生支持非线性数据的处理
- 较难处理数据不平衡问题

## 损失函数

logistic regression 中不再沿用 linear regression 中的最小二乘作为损失函数，原因如下：

1. 假设有三个样本，类别分别为 1, 1, 0，模型 1 预测结果为0.6 , 0.6, 0.4，模型 2 预测结果为 0.9, 0.9, 0.6，由于模型 1 全部分类正确，而模型 2 分错了一个，前者优于后者。但是模型 1 的最小二乘为0.16 + 0.16 + 0.16 = 0.48，模型 2 的最小二乘为 0.01 + 0.01 + 0.36 = 0.38，后者却小于前者，与实际的结论产生矛盾。
2. 若用最小二乘作为损失函数，<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{2}\sum_{i&space;=&space;0}^n&space;(\frac{1}{1&plus;e^{-\theta^T&space;\cdot&space;x&space;}}&space;-&space;y_i)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{2}\sum_{i&space;=&space;0}^n&space;(\frac{1}{1&plus;e^{-\theta^T&space;\cdot&space;x&space;}}&space;-&space;y_i)^2" title="\frac{1}{2}\sum_{i = 0}^n (\frac{1}{1+e^{-\theta^T \cdot x }} - y_i)^2" /></a> 包含了 sigmoid 函数，不是凸函数，所以没有最优解。

由于每个样本都是独立的，所以该样本集的联合概率分布为：

<img src="https://latex.codecogs.com/gif.latex?\prod&space;_{i=1}^mp^{y_i}&space;\cdot&space;(1&space;-&space;p)^{1-y_i}" title="\prod _{i=1}^mp^{y_i} \cdot (1 - p)^{1-y_i}" />

将 <img src="https://latex.codecogs.com/gif.latex?p&space;=&space;\frac{1}{1&space;&plus;&space;e^{-\theta^T&space;\cdot&space;x&space;}}" title="p = \frac{1}{1 + e^{-\theta^T \cdot x }}" /> 代入上式，并用极大似然估计法求其最大值时的 <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /> 的值：

<img src="https://latex.codecogs.com/gif.latex?\prod&space;_{i=1}^m(\frac{1}{1&space;&plus;&space;e^{-\theta^T&space;\cdot&space;x&space;}})^{y_i}&space;\cdot&space;(\frac{e^{-\theta^T&space;\cdot&space;x&space;}}{1&space;&plus;&space;e^{-\theta^T&space;\cdot&space;x&space;}})^{1-y_i}" title="\prod _{i=1}^m(\frac{1}{1 + e^{-\theta^T \cdot x }})^{y_i} \cdot (\frac{e^{-\theta^T \cdot x }}{1 + e^{-\theta^T \cdot x }})^{1-y_i}" />

为了便于计算，取自然对数变成加和形式：

<img src="https://latex.codecogs.com/gif.latex?\sum&space;_{i=0}^{m}y_iln(\frac{1}{1&plus;e^{-\theta^T&space;\cdot&space;x}})&space;&plus;&space;(1-y_i)ln(\frac{e^{-\theta^T&space;\cdot&space;x}}{1&plus;e^{-\theta^T&space;\cdot&space;x}})" title="\sum _{i=0}^{m}y_iln(\frac{1}{1+e^{-\theta^T \cdot x}}) + (1-y_i)ln(\frac{e^{-\theta^T \cdot x}}{1+e^{-\theta^T \cdot x}})" />

<img src="https://latex.codecogs.com/gif.latex?\sum&space;_{i=0}^{m}y_i(\theta^T&space;\cdot&space;x)&space;&plus;&space;ln(1&space;&plus;&space;e^{\theta^T&space;\cdot&space;x})" title="\sum _{i=0}^{m}y_i(\theta^T \cdot x) + ln(1 + e^{\theta^T \cdot x})" />

对 <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /> 求偏导得到梯度：

<img src="https://latex.codecogs.com/gif.latex?\sum&space;_{i=0}^{m}(y_i-\frac{1}{1&plus;e^{-\theta^T&space;\cdot&space;x}})x_i" title="\sum _{i=0}^{m}(y_i-\frac{1}{1+e^{-\theta^T \cdot x}})x_i" />

极大似然估计求概率最大时的参数值，因此是梯度上升的方向，取负号便是梯度下降的方向：

<div  align="center"> 
<img src="https://latex.codecogs.com/gif.latex?-\sum&space;_{i=0}^{m}(y_i-\frac{1}{1&plus;e^{-\theta^T&space;\cdot&space;x}})x_i" title="-\sum _{i=0}^{m}(y_i-\frac{1}{1+e^{-\theta^T \cdot x}})x_i" />
</div> 

## 多分类

1. one vs all
2. one vs one
3. softmax
