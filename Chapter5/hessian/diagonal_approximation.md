上面讨论的Hessian矩阵的一些应用需要求出Hessian矩阵的逆矩阵，而不是Hessian矩阵本身。因此，我们对Hessian矩阵的对角化近似比较感兴趣。换句话说，就是把非对角线上的元素置为零，因为这样做之后，矩阵的逆很容易计算。与之前一样，我们考虑由一系列项的求和项组成的误差函数，每一项对应于数据集里的一个模式，即$$ E = \sum_n E_n $$。这样，Hessian矩阵可以通过每次考虑一个模式然后对所有模式求和的方法得到。根据式（5.48），对于模式$$ n $$，Hessian矩阵的对角线元素可以写成    

$$
\frac{\partial^2 E_n}{\partial w_{ji}^2} = \frac{\partial^2 E_n}{\partial a_j^2}z_j^2 \tag{5.79}
$$

使用式（5.48）（5.49），式（5.79）右手边的二阶导数可以通过递归地使用微分的链式法则来给出形式为

$$
\frac{\partial^2 E_n}{\partial a_j^2} = h'(a_j)^2\sum\limits_k\sum\limits_{k'}w_{kj}w_{k'j}\frac{\partial^2E_n}{\partial a_k\partial a_{k'}} + h''(a_j)\sum\limits_kw_{kj}\frac{\partial E_n}{\partial a_k} \tag{5.80}
$$

的反向传播方程来获得。如果我们忽略二阶导数中非对角线元素，那么我们有(Becker and LeCun, 1989; LeCun et al., 1990) 

$$
\frac{\partial^2 E_n}{\partial a_j^2} = h'(a_j)^2\sum\limits_kw_{kj}^2\frac{\partial^2E_n}{\partial a_k^2} + h''(a_j)\sum\limits_kw_{kj}\frac{\partial E_n}{\partial a_k} \tag{5.81}
$$

注意，计算这个近似所需的计算步骤数为$$ O(W) $$，其中$$ W $$是网络中权值和偏置的总数。对比全Hessian矩阵的计算步骤数$$ O(W^2) $$。    


Ricotti et al.(1998)也使用了Hessian矩阵的对角近似，但是他们在计算$$ \partial^2 E_n/\partial a_j^2 $$时保留了所有项，从而得到了对角项的精确的表达式。注意，这样就不再具有$$ O(W) $$的计算复杂度。然而，对角近似的主要问题是在实际应用中Hessian矩阵通常是强烈非对角化的，所以为了计算方便而采取的这些近似手段必须非常谨慎。     

