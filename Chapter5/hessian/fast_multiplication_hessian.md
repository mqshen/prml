对于Hessian矩阵的许多应用来说，我们感兴趣的不是Hessian矩阵$$ H $$本身，而是$$ H $$与某些向量$$ v $$的乘积。我们已经知道Hessian矩阵的计算需要$$ O(W^2) $$次操作，所需的存储空间也是$$ O(W^2) $$。但是，我们想要计算的向量$$ v^TH $$只有$$ W $$个元素。因此，我们可以不把计算Hessian矩阵当成一个中间的步骤，而是尝试寻找一种只需$$ O(W) $$次操作的高效方法来直接计算$$ v^TH $$。    

为了达到这个目的，我们首先注意到

$$
v^TH = v^T\nabla(\nabla E) \tag{5.96}
$$

其中$$ \nabla $$表示权空间的梯度。然后，我们可以写下计算$$ \nabla E $$的标准正向传播和反向传播的方程，然后对这些方程应用式（5.96）得到一组计算$$ v^TH $$的正向传播和反向传播的方程（Møller, 1993; Pearlmutter, 1994）。这对应于将微分操作$$ v^T\nabla $$作用于原始的正向传播和反向传播的方程。Pearlmutter(1994)使用记号$$ R\{\dot\} $$表示操作符$$ v^T\nabla
$$，我们将遵从这个惯例。分析过程很直接，我们会使用微积分的通用规则，以及

$$
R\{w\} = v \tag{5.97}
$$

这个结果。    

我们会使用一个简单的例子来很好的说明这种技术。同样的，我们使用图5.1展示的两层网络，及线性的输出单元和平方和误差函数。与之前一样，我们考虑数据集里的一个模式对于误差函数的贡献。这样，我们所要求的向量可以通过每个模式各自的贡献然后求和得到。对于两层神经网络，正向传播方程由    

$$
\begin{eqnarray}
a_j &=& \sum\limits_i w_{ji}x_i \tag{5.98} \\
z_j &=& h(a_j) \tag{5.99} \\
y_k &=& \sum\limits_j w_{kj}z_j \tag{5.100}
\end{eqnarray}
$$

给出。在这些方程上运用$$ R\{\dot\} $$，得到一组形式为

$$
\begin{eqnarray}
R\{a_j\} &=& \sum\limits_i v_{ji}x_i \tag{5.101} \\
R\{z_j\} &=& h'(a_j)R\{a_j\} \tag{5.102} \\
R\{a_j\} &=& \sum\limits_jw_{kj}R\{z_j\} + \sum\limits_j v_{kj}z_j \tag{5.103}
\end{eqnarray}
$$

其中，$$ v_{ji} $$是向量$$ v $$中对应于权值$$ w_\{ji\} $$的元素。$$ R\{z_j\}, R\{a_j\}, R\{y_k\} $$可以被看成新变量，它的值可以使用上面的方程得到。    

由于我们正在考虑平方和误差函数，因此我们得到标准的反向传播表达式：    

$$
\begin{eqnarray}
\delta_k &=& y_k - t_k \tag{5.104} \\
\delta_j &=& h'(a_j)\sum\limits_k w_{kj}\delta_k \tag{5.105}
\end{eqnarray}
$$

同样的，在这些方程上运用$$ R\{\dot\} $$，得到一组形式为

$$
\begin{eqnarray}
R\{\delta_k\} &=& R\{y_k\} \tag{5.106} \\
R\{\delta_j\} &=& h''(a_j)R\{a_j\}\sum\limits_kw_{kj}\delta_k \\
& & + h'(a_j)\sum\limits_kv_{kj}\delta_k + h'(a_j)\sum\limits_kw_{kj}R\{\delta_k\} \tag{5.107}
\end{eqnarray}
$$

最后，我们有误差函数的一阶导数的方程：

$$
\begin{eqnarray}
\frac{\partial E}{\partial w_{kj}} &=& \delta_kz_j \tag{5.108} \\
\frac{\partial E}{\partial w_{ji}} &=& \delta_jx_i \tag{5.109}
\end{eqnarray}
$$


在这些方程上运用$$ R\{\dot\} $$，得到$$ v^TH $$的元素的表达式：    

$$
\begin{eqnarray}
R\left\{\frac{\partial E}{\partial w_{kj}}\right\} &=& R\{\delta_k\}z_j + \delta_kR\{z_j\} \tag{5.110} \\
R\left\{\frac{\partial E}{\partial w_{ji}}\right\} &=& x_iR\{\delta_j\} \tag{5.111}
\end{eqnarray}
$$

算法的实现涉及将新变量$$ R\{a_j\}, R\{z_j\}, R\{\delta_j\} $$引入到隐藏单元，并将$$ R\{\delta_k\}, R\{y_k\} $$引入到输出单元。对于每个输入模式，这些量的值可以使用之前的结果求出，$$ v^TH $$的元素的值由式（5.110）和式（5.111）给出。这种方法的一个好处是，计算$$ v^TH $$的方程与标准的正向传播和反向传播的方程相同，因此将现有软件扩展到能够计算这个乘积通常很容易。    

如果必要的话，这个方法可以用来计算完整的Hessian矩阵。计算的方法为：将向量$$ v $$选为一系列的形如$$ (0, 0,...,1,...,0) $$的单位向量，每个单位向量选出Hessian矩阵中的一列。这种方法的数学形式与Bishop(1992)的反向传播算法等价，如5.4.5节所述。但是由于这种方法存在冗余计算，会损失一定的计算效率。


