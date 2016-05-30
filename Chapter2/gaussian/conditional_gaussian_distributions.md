多元高斯分布的一个重要的性质是：如果两个变量集合是联合高斯分布，以其中一个集合为条件的分布也是高斯分布。同样的，任何一个变量的边缘分布也是高斯分布。    

首先，考虑条件概率的情形。假设$$ x $$是服从高斯分布$$ \mathcal{N}(x|\mu, \Sigma) $$的$$ D $$维向量，把$$ x $$划分为两个不相交的子集$$ x_a, x_b $$。不失一般性的，令$$ x_a $$为$$ x $$的前$$ M $$个分量，令$$ x_b $$为剩余的$$ D − M $$个分量，得到：

$$ 
x = 
\left(
\begin{array}{c}  
x_a \\
x_b 
\end{array}
\right) \tag{2.65}
$$

对应的均值向量$$ \mu $$的划分：

$$ 
\mu = 
\left(
\begin{array}{c}  
\mu_a \\
\mu_b 
\end{array}
\right) \tag{2.66}
$$

协方差矩阵为：

$$ 
\Sigma = 
\left(
\begin{array}{cc}  
\Sigma_{aa} & \Sigma_{ab} \\
\Sigma_{ba} & \Sigma_{bb} \\
\end{array}
\right) \tag{2.67}
$$

注意，协方差矩阵是对称的即$$ \Sigma^T = \Sigma $$，可得$$ \Sigma_{aa},\Sigma_{bb} $$也是对称的，且$$ \Sigma_{ba} = \Sigma_{ab}^T $$。

在很多情况下，使用协方差的逆矩阵会比较方便，记：

$$
\Lambda \equiv \Sigma^{-1} \tag{2.68}
$$

这被称为精度矩阵（precision matrix）。事实上，高斯分布的一些形式使用协方差来表达会比较方便，而另外的一些使用精度矩阵会比较方便。以式（2.65）划分$$ x $$的同样的方法划分精度矩阵：    

$$
\Lambda = 
\left(
\begin{array}{cc}  
\Lambda_{aa} & \Lambda_{ab} \\
\Lambda_{ba} & \Lambda_{bb} \\
\end{array}
\right) \tag{2.69}
$$

因为对称矩阵的逆同样是对称的，所以$$ \Lambda_{aa},\Lambda_{bb} $$也是对称的，且$$ \Lambda_{ab}^T = \Lambda_{ba} $$。需要强调的一点是：事实上，$$ \Lambda_{aa} $$不单单是对$$ \Sigma_{aa} $$求逆这么简单。稍后，就会讨论逆矩阵的划分和划分的逆矩阵间的关系。    

首先，找到条件分布$$ p(x_a|x_b) $$的条件分布。根据概率的乘法规则，由联合分布 $$ p(x) = p(x_a, x_b) $$通过固定$$ x_b $$为观测到的值，然后标准化所得到的表达式就可以得到$$ x_a $$上的有效概率。考虑公式（2.44）给出的高斯分布的指数上的二项式，在计算的最后阶段再来考虑标准化系数，这样比显示地进行标准化更有效。使用公式（2.65），（2.66）和（2.69）的划分就能得到：    

$$
\begin{eqnarray}
&-\frac{1}{2}&(x-\mu)^T\Sigma^{-1}(x-\mu) = \\
&-\frac{1}{2}&(x_a - \mu_a)^T\Lambda_{aa}(x_a - \mu_a) -\frac{1}{2}(x_a - \mu_a)^T\Lambda_{ab}(x_b - \mu_b) \\
&-\frac{1}{2}&(x_b - \mu_b)^T\Lambda_{ba}(x_a - \mu_a) -\frac{1}{2}(x_b - \mu_b)^T\Lambda_{bb}(x_b - \mu_b) \tag{2.70}
\end{eqnarray}
$$

把它看成$$ x_a $$的函数，这又是一个二次型，可以推出对应的条件分布$$ p(x_a|x_b) $$是高斯分布。由于，这个分布完全是由均值和方差定义的，我们的目标是通过式（2.70）来确定$$ p(x_a|x_b) $$的均值和方差的表达式。    

给出一个定义高斯分布的指数项的二次型，然后确定对应的均值和方差的方法被称为“配平方”，这是一种与高斯分布相关的常见的操作。一个通用的的高斯分布$$ \mathcal{N}(x|\mu, \Sigma) $$的指数项可以写成：    

$$
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) = -\frac{1}{2}x^T\Sigma^{-1}x + x^T\Sigma^{-1}\mu + const \tag{2.71}
$$

其中const为与$$ x $$无关的项，且用到的$$ \Sigma $$的对称性。因此，如果我们把通用的二次型表示成公式（2.71）右侧的形式，那么我们可以立即令$$ x $$中的二阶项的系数矩阵等于协方差矩阵的逆$$ \Sigma^{−1} $$，令$$ x $$中的线性项的系数等于$$ \Sigma^{-1}\mu $$，这样就得到了$$ \mu $$。    

现在让我们把这个方法应用到条件高斯分布$$ p(x_a|x_b) $$中。其中条件高斯分布的指数项的二次型由公式（2.70）给出。把这个分布的均值和协方差分别记作$$ \mu_{a|b}, \Sigma_{a|b} $$。考虑公式（2.70）对$$ x_a $$的函数依赖关系，其中$$ x_b $$被当成常数。如果我们选出所有$$ x_a $$的二阶项，就有：

$$ -\frac{1}{2}x_a^T\Lambda_{aa}x_a \tag{2.72} $$

从这个公式中可以得到，$$ p(x_a|x_b) $$的协方差（精度矩阵的逆）：    

$$ \Sigma_{a|b} = \Lambda_{aa}^{-1} \tag{2.73} $$

现在，考虑式（2.70）中$$ x_a $$的所有线性项：

$$ x_a^T\left\{ \Lambda_{aa}\mu_a - \Lambda_{ab}(x_b - \mu_b)\right\} \tag{2.74} $$

其中我们使用了$$ \Lambda_{ba}^T = \Lambda_{ab} $$。根据通用公式（2.71）这个表达始中$$ x_a $$的系数必须等于$$ \Sigma_{a|b}^{-1}\mu_{a|b} $$，推出：    

$$
\begin{eqnarray}
\mu_{a|b} &=& \Sigma_{a|b}\left\{\Lambda_{aa}\mu_a - \Lambda_{ab}(x_b - \mu_b)\right\} \\
&=& \mu_a - \Lambda_{aa}^{-1}\Lambda_{ab}(x_b - \mu_b) \tag{2.75}
\end{eqnarray}
$$

过程中我们使用了式（2.73）    


式（2.73），（2.75）的结果是由原始的联合分布$$ p(x_a, x_b) $$的分区精度矩阵来表示的。也可以用对应的分区协方差矩阵来表达这些结果。为了完成这一点，使用下面的关于分区矩阵的逆的等式：    

$$
\left(
\begin{array}{cc}  
A & B \\
C & D 
\end{array}
\right)^{-1}
= 
\left(
\begin{array}{cc}  
M & -MBD^{-1} \\
-D^{-1}CM & D^{-1} + D^{-1}CMBD^{-1} 
\end{array}
\right) \tag{2.76}
$$

其中$$ M = (A - BD^{-1}C)^{-1} \tag{2.77} $$。

$$ M^{-1} $$是式（2.76）左手边矩阵关于矩阵$$ D $$的舒尔补（Schur complement）。使用定义

$$
\left(
\begin{array}{cc}  
\Sigma_{aa} & \Sigma_{ab} \\
\Sigma_{ba} & \Sigma_{bb} \\
\end{array}
\right)^{-1}
=
\left(
\begin{array}{cc}  
\Lambda_{aa} & \Lambda_{ab} \\
\Lambda_{ba} & \Lambda_{bb} \\
\end{array}
\right) \tag{2.78}
$$

使用公式（2.76）可得：

$$
\begin{eqnarray}
\Lambda_{aa} &=& (\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba} )^{-1} \tag{2.79} \\
\Lambda_{ab} &=& -(\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba} )^{-1}\Sigma_{ab}\Sigma_{bb}^{-1} \tag{2.80}
\end{eqnarray}
$$

从这些公式中，得到条件概率$$ p(x_a|x_b) $$的均值和方差的表达式为：
$$
\begin{eqnarray}
\mu_{a|b} &=& \mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b) \tag{2.81} \\
\Sigma_{a|b} &=& \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba} \tag{2.82}
\end{eqnarray}
$$

对比式（2.73）和（2.82），得到条件概率分布$$ p(x_a|x_b) $$如果使用分区精度矩阵而不是分区协方差矩阵表示，那么它的形式会更简单。注意，条件概率分布$$ p(x_a|x_b) $$的由式（2.81）给出的均值是$$ x_b $$的线性函数，由公式（2.82）给出的协方差与$$ x_a $$无关。这是线性高斯（linear-Gaussian）模型的一个例子。
