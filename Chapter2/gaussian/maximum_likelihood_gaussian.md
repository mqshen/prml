给定一个数据集 $$ X = (x_1,...,x_N)^T $$，其中假定观测$$ \{x_n\} $$是独立地从多元高斯分布中抽取的，我们可以使用最大似然来估计分布的参数。对数似然函数为：    

$$
\ln p(X|\mu, \Sigma) = -\frac{ND}{2}\ln(2\pi)-\frac{N}{2}\ln |\Sigma| - \frac{1}{2}\sum\limits_{n=1}^{N}(x_n - \mu)^T\Sigma^{-1}(x_n - \mu) \tag{2.118}
$$

通过简单的从新排列，得到最大似然函数只依赖于数据集的两个量：    

$$ \sum\limits_{n=1}^Nx_n , \sum\limits_{n=1}^Nx_nx_n^T \tag{2.119} $$ 

这些被称为高斯分布的充分统计量（sufficient statistics）。使用式（C.19）对数似然关于$$ \mu $$的导数为：    

$$
\frac{\partial}{\partial\mu}\ln p(X|\mu,\Sigma) = \sum\limits_{n=1}^N\Sigma^{-1}(x_n - \mu) \tag{2.120}
$$

并设导数为0，得到了均值的最大似然估计为：    

$$
\mu_{ML} = \frac{1}{N}\sum\limits_{n=1}^Nx_n \tag{2.121}
$$

这是数据观测集合的均值。式（2.118）关于$$ \Sigma $$的最大化会比较复杂。最简单的方法是忽略对称性的约束，然后证明结果像我们要求的那样对称的。另一种推导方式是显示的利用对称性和正定性约束，就像Magnus and Neudecker (1999)那样。符合预期的结果的形式为：    

$$
\Sigma_{ML} = \frac{1}{N}\sum\limits_{n=1}^N(x_n - \mu_{ML})(x_n - \mu_{ML})^T \tag{2.122}
$$

结果涉及到了$$ \mu_{ML} $$，因为这是关于$$ \mu, \Sigma $$的联合最大值的结果。注意，式（2.121）的$$ \mu_{ML} $$的结果不依赖于$$ \Sigma_{ML} $$，所以我们可以先估计$$ \mu_{ML} $$在利用这个结果得到$$ \Sigma_{ML} $$。    

如果我们估计真实分布下的最大似然期望，可以得到：    

$$
\begin{eqnarray}
\mathbb{E}[\mu_{ML}] &=& \mu \tag{2.123} \\
\mathbb{E}[\Sigma_{ML}] &=& \frac{N - 1}{N}\Sigma \tag{2.124}
\end{eqnarray}
$$    

得到最大似然估计的均值的期望等于真实的均值。但是，它对于协方差的估计的期望小于真实的方差，所以这是有偏的。我们可以定义一个不同的估计值    

$$
\widetilde{\Sigma} = \frac{1}{N-1}\sum\limits_{n=1}^N(x_n - \mu_{ML})(x_n - \mu_{ML})^T \tag{2.125}
$$

通过式（2.122），（2.124）可以显然的得到$$ \widetilde{\Sigma} $$的期望等于$$ \Sigma $$。
