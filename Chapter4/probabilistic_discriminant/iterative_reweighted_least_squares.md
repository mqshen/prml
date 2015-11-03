在第三章中讨论的线性回归模型中，基于高斯噪声模型假设的最大似然方法是有解析解的。这是由于对数似然函数是参数向量$$ w $$的二次函数。对于logistic回归来说，由于logistic
sigmoid函数是一个非线性函数，所以它不再有解析解。然而不再是二次形式并不是本质原因。准确的说，正如我们看到的那样，误差函数是凸的，因此有一个唯一的最小值。此外，误差函数可以通过基于Newton-Raphson迭代最优化模式来得到对数似然函数的局部二次近似的高效迭代算法来求出最小值。为了得到函数$$ E(w) $$的最小值，Newton-Raphson方法使用

$$
w^{(new)} = w^{(old)} - H^{-1}\nabla E(w) \tag{4.92}
$$

这个等式来更新权值，其中$$ H $$是由$$ E(w) $$关于$$ w $$的二阶导数组成Hessian矩阵。    

我们先对线性回归模型（3.3）的平方和误差函数（3.12）使用Newton-Raphson方法。这个误差函数的梯度和Hessian矩阵由    

$$
\begin{eqnarray}
\nabla E(w) = \sum\limits_{n=1}^N(w^T\phi_n-t_n)\phi_n = \Phi^T\Phi w - \Phi^Tt \tag{4.93} \\
H = \nabla\nabla E(w) = \sum\limits_{n=1}^N\phi_n\phi_n^T = \Phi^T\Phi \tag{4.94}
\end{eqnarray}
$$

其中$$ \Phi $$是$$ N \times M $$的设计矩阵，它的第$$ n^{th} $$行由$$ \phi_n^T $$给出。这样Newton-Raphson的更新迭代式就变成了

$$
\begin{eqnarray}
w^{(new)} &=& w^{(old)} - \left(\Phi^T\Phi\right)^{-1}\left\{\Phi^T\Phi w^{(old)} - \Phi^Tt\right\} \\
&=& \left(\Phi^T\Phi\right)^{-1}\Phi^Tt \tag{4.95}
\end{eqnarray}
$$    

这就是我们所说的标准最小二乘解。注意，这种情况下的误差函数是二次的，所以Newton-Raphson公式一步就能给出精确解。    

现在，我们在logistic回归模型的交叉熵误差函数（4.90）上运用Newton-Raphson更新。从式（4.91）可以得到，这个误差函数的梯度和Hessian由形式

$$
\begin{eqnarray}
\nabla E(w) &=& \sum\limits_{n=1}^N(y_n-t_n)\phi_n = \Phi^T(y-t) \tag{4.96} \\
H &=& \nabla\nabla E(w) = \sum\limits_{n=1}^Ny_n(1-y_n)\phi_n\phi_n^T = \Phi^TR\Phi \tag{4.97}
\end{eqnarray}
$$

其中我们使用了（4.88）。也引入了元素为

$$
R_{nn} = y_n(1-y_n) \tag{4.98}
$$

的$$ N \times N $$的对角矩阵$$ R $$。我们看到Hessian矩阵不再是常量，而是通过权矩阵$$ R $$依赖于$$ w $$。这对应误差函数不是二次函数的事实。使用来自与logistic sigmoid函数的性质$$ 0 < y_n < 1 $$，得到对于任意向量$$ u $$都有$$ u^THu > 0 $$，所以Hessian矩阵$$ H $$是正定的。所以误差函数是一个关于$$ w $$的凸函数，因此有唯一的最小值。    

然后，logistic回归模型的Newton-Raphson更新公式就变成了

$$
\begin{eqnarray}
w^{(new)} &=& w^{(old)} - (\Phi^TR\Phi)^{-1}\Phi^T(y-t) \\
&=& (\Phi^TR\Phi)^{-1} \left\{(\Phi^TR\Phi)^{-1}w^{(old)}-\Phi^T(y-t)\right\} \\
&=& (\Phi^TR\Phi)^{-1}\Phi^TRz \tag{4.99}
\end{eqnarray}
$$

其中$$ z $$是元素为

$$
z = \Phi w^{(old)} - R^{-1}(y-t) \tag{4.100}
$$

的$$ N $$维矩阵。我们看到公式（4.99）为一组加权最小二乘问题的标准方程。由于权重矩阵$$ R $$不是常数，而是依赖参数向量$$ w $$的。因此我们必须使用标准方程来迭代计算，每次使用新的权向量$$ w $$来修正权重矩阵$$ R $$。由于这个原因，这个算法被称为迭代再加权最小二乘（iterative reweighted least squares）或IRLS（Rubin, 1983）。与加权的最小二乘问题一样，由于$$ t $$的均值和方差由

$$
\begin{eqnarray}
\mathbb{E}[t] &=& \sigma(x) = y \tag{4.101} \\
var[t] &=& \mathbb[t^2] - \mathbb[t]^2 = \sigma(x) - \sigma(x)^2 = y(1-y) \tag{4.102}
\end{eqnarray}
$$

给出，其中使用了性质$$ t^2 = t, t \in \{0,1\} $$，所以可以把对角权重矩阵$$ R $$的元素看成方差。实际上，我们可以把IRLS解释成变量$$ a = w^T\phi $$空间中的线性化问题的解。这样$$ z $$的第$$ n^{th} $$个元素$$ z_n $$的值可以简单的解释为通过对logistic sigmoid函数在当前操作点$$ w^{(old)} $$周围局部线性近似得到的这个空间中的有效的目标值。    

$$
\begin{eqnarray}
a_n(w) &\simeq& a_n\left(w^{(old)}\right) + \left.\frac{da_n}{dy_n}\vphantom{\Big|}\right| _{w^{(old)}}(t_n-y_n) \\
&=& \phi_n^Tw^{(old)} - \frac{(y_n-t_n)}{y_n(1-y_n)} = z_n \tag{4.103}
\end{eqnarray}
$$

