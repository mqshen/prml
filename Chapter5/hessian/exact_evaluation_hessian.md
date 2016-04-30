目前为止，我们已经考察了各种计算Hessian矩阵或逆矩阵的近似方法。对于一个任意的前馈拓扑结构的网络，Hessian矩阵也可以使用反向传播算法计算一阶导数的推广来精确的计算。这同时也保留了计算一阶导数的方法的许多良好的性质，包括计算效率(Bishop, 1991; Bishop, 1992)。这种方法可以应用于任何表示网络输出的的可微的误差函数，及任何具有可微的激活函数的神经网络。计算Hessian矩阵的复杂度为$$ O(W^2) $$。类似的算法也可以参考Buntine and Weigend(1993)。    

这里我们考虑具有两层权值（待求的方程很容易推导）的网络。我们将使用下标$$ i, i' $$表示输入，用下标$$ j, j' $$表示隐藏单元，用下标$$ k, k' $$表示输出。首先定义     

$$
\delta_k = \frac{\partial E_n}{\partial a_k}  , M_{kk'} \equiv \frac{\partial^2 E_n}{\partial a_k\partial a_{k'}} \tag{5.92}
$$

其中$$ E_n $$表示数据点$$ n $$对误差的贡献。这个网络的Hessian矩阵可以被看成三个独立的项：

1. 两个权值都在第二层
    
    $$
    \frac{\partial^2 E_n}{\partial w_{kj}^{(2)}\partial w_{k'j'}^{(2)}} = z_jz_{j'}M_{kk'} \tag{5.93}
    $$
    
2. 两个权值都在第一层    
    
    $$
    \begin{eqnarray}
    \frac{\partial^2 E_n}{\partial w_{ji}^{(1)}\partial w_{j'i'}^{(1)}} = x_ix_{i'}h''(a_{j'})I_{jj'}\sum\limits_kw_{kj'}^{(2)}\delta_k \\
    + x_ix_{i'}h'(a_{j'})h'(a_j)\sum\limits_k\sum\limits_{k'}w_{k'j'}^{(2)}w_{kj}^{(2)}M_{kk'} \tag{5.94}
    \end{eqnarray}
    $$
    
3. 权值分别在两层    

    $$
    \frac{\partial^2 E_n}{\partial w_{ji}^{(1)}\partial w_{kj'}^{(2)}} = x_ih'(a_{j'})\left\{\delta_kI_{jj'} + z_j\sum\limits_{k'}w_{k'j'}^{(2)}H_{kk'}\right\} \tag{5.95}
    $$

这里$$ I_{jj'} $$是单位矩阵的第$$ j, j' $$个元素。如果权值中的一个或两个是偏置项，那么只需将激活设为1就可以得到对应的表达式。很容易将这个结果推广到网络包含跨层连接的情形。
