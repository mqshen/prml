目前为止，在本章中学习的概率分布（除了高斯混合分布）都是一种叫做指数族这一大类分布中的特殊例子（Duda and Hart, 1973; Bernardo and Smith, 1994）。指数族分布的成员拥有很多共同的重要性质，且在某种程度的通用性下讨论这些性质很有启发性。    

给定参数$$ \eta $$的$$ x $$上的指数族分布是具有

$$
p(x|\eta) = h(x)g(\eta)exp\{\eta^Tu(x)} \tag{2.194}
$$

形式的概率分布的集合。其中$$ x $$可以是标量也可以是向量，可以是连续的也可以是离散的。$$ \eta $$是分布的自然参数（natural parameters），$$ u(x) $$是关于$$ x $$的某个函数。函数$$ g(\eta) $$可以解释为是为了保证分布标准化的系数，且满足：    

$$
g(\eta)\int h(x)exp\{\eta^Tu(x)\}dx = 1 \tag{2.195}
$$

其中，对于离散变量积分就变成求和。    

首先，给出一些本章之前讨论的一些分布，然后证明这些分布确实是指数族分布。首先考虑伯努利分布：    

$$
p(x|\mu) = Bern(x|\mu) = \mu^x(1-\mu)^{1-x} \tag{2.196}
$$

把右侧表示成对数的指数形式，得到：    

$$
\begin{eqnarray}
p(x|\mu) &=& exp\{x\ln \mu + (1-x)\ln (1-\mu)\} \\
&=& (1-\mu)exp\left\{\ln\left(\frac{\mu}{1-\mu}\right)x\right\} \tag{2.197}
\end{eqnarray}
$$

与公式（2.194）对照，得到：    

$$
\eta = \ln\left(\frac{\mu}{1-\mu}\right) \tag{2.198}
$$

然后就可以解出$$ \mu = \delta (\eta) $$，其中    

$$
\delta (\eta) = \frac{1}{1+exp(-\eta)} \tag{2.199}
$$

这就是logistic sigmoid函数。因此可以把伯努利分布写成式（2.194）的标准形式：    

$$
p(x|\eta) = \delta(-\eta)exp(\eta x) \tag{2.200}
$$

其中使用了可以从式（2.199）中很容易证明的$$ 1 - \delta(\eta) = \delta(-\eta) $$，对比公式（2.194）得到：    

$$
\begin{eqnarray}
u(x) = x \tag{2.201} \\
h(x) = 1 \tag{2.202} \\
g(\eta) = \delta(-\eta) \tag{2.203}
\end{eqnarray}
$$

接下来，考虑单观测值$$ x $$的多项式分布：    

$$
p(x|\mu) = \prod\limits_{k=1}^M\mu_k^{x_k} = exp\left\{\sum\limits_{k=1}^M x_k \ln\mu_k \right\} \tag{2.204}
$$

其中$$ x = (x_1,...,x_N)^T $$。同样的，可以写成式（2.194）的标准形式：    

$$
p(x|\eta) = exp(\eta^Tx) \tag{2.205}
$$

其中$$ \eta_k = \ln \mu_k $$，且定义了$$ \eta = (\eta_1,...,\eta_M)^T $$。同样，对比式（2.194）得到：    

$$
\begin{eqnarray}
u(x) = x \tag{2.206} \\
h(x) = 1 \tag{2.207} \\
g(\eta) = 1 \tag{2.208}
\end{eqnarray}
$$

注意，因为参数$$ \mu_k $$要满足

$$
\sum\limits_{k=1}^M\mu_k = 1 \tag{2.209}
$$

，所以给定任意$$ M − 1 $$个参数$$ \mu_k $$剩下的参数就固定了，因此参数$$ \eta_k $$不是相互独立的。在某些情况下，去掉这个限制，只用$$ M − 1 $$个参数来表示分布会比较方便。可以使用式（2.209）中的关系，用$$ \{\mu_k\}，k=1,...,M-1 $$来表示最后的$$ \mu_M $$，这样就只剩下$$ M - 1 $$个参数了。注意，剩余的参数仍然要满足：    

$$
0 \leq \mu_k \leq 1, \sum\limits_{k=1}^{M-1}\mu_k \leq 1 \tag{2.210}
$$

使用式（2.209）的约束，这种表达方式下多项式分布变成：

$$
\begin{eqnarray}
&exp&\left\{\sum\limits_{k=1}^Mx_k\ln \mu_k \right\} \\
&=& exp\left\{\sum\limits_{k=1}^{M-1}x_k\ln\mu_k + \left(1-\sum\limits_{k=1}^{M-1}x_k\right)\ln\left(1-\sum\limits_{k=1}^{M-1}\mu_k\right)\right\} \\
&=& exp\left\{\sum\limits_{k=1}^{M-1}x_k\ln\left(\frac{\mu_k}{1-\sum_{j=1}^{M-1}\mu_j}\right) + \ln\left(1-\sum\limits_{k=1}^{M-1}\mu_k\right)\right\} \tag{2.211}
\end{eqnarray}
$$

现在，确定

$$
\ln\left(\frac{\mu_k}{1-\sum_j\mu_j}\right)=\eta_k \tag{2.212}
$$

首先两边对$$ k $$求和，然后重新整理，回带，就可以解出$$ \mu_k $$：    

$$
\mu_k = \frac{exp(\eta_k)}{1+\sum_jexp(\eta_j)} \tag{2.213}
$$

这被称为softmax函数，或标准化指数（normalized exponential）。在这种表达方式下，多项式分布具有：    

$$
p(x|\eta) = \left(1 + \sum\limits_{k=1}^{M-1}exp(\eta_k)\right)^{-1} exp(\eta^Tx) \tag{2.214}
$$

这是具有参数向量$$ \eta = (\eta_1,...,\eta_{M-1})^T $$的指数族的标准形式。其中：

$$
\begin{eqnarray}
u(x) = x \tag{2.215} \\
h(x) = 1 \tag{2.216} \\
g(\eta) = \left(1+\sum\limits_{k=1}^{M-1}exp(\eta_k)\right)^{-1} \tag{2.217} 
\end{eqnarray}
$$

最后，考察高斯分布。对于一元高斯有：    

$$
\begin{eqnarray}
p(x|\mu,\delta^2) &=& \frac{1}{(2\pi\delta^2)^{1/2}}exp\left\{-\frac{1}{2\delta^2}(x-\mu)^2\right} \tag{2.218} \\
&=& \frac{1}{(2\pi\delta^2)^{1/2}}exp\left\{-\frac{1}{2\delta^2}x^2 + \frac{\mu}{\delta^2}x - \frac{1}{2\delta^2}\mu^2 \right} \tag{2.219}
\end{eqnarray}
$$

经过一些简单的重排列之后，可以转化为式（2.194）给出的标准指数族分布的形式，其中：

$$
\begin{eqnarray}
\eta &=& \left( \begin{array}{c} \mu/\delta^2 \\ -1/2\delta^2 \end{array} \right) \tag{2.220}  \\
u(x) &=& \left( \begin{array}{c} x \\ x^2 \end{array} \right) \tag{2.221}  \\
h(x) &=& (2\pi)^{-1/2} \tag{2.222} \\
g(\eta) &=& = (-2\eta_2)^{1/2}exp\left(\frac{\eta_1^2}{4\eta_2}\right) \tag{2.223}
\end{eqnarray}
$$

