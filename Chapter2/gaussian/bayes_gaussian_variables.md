在2.3.1和2.3.2中，我们讨论了把高斯分布$$ p(x) $$的向量$$ x $$切分成$$ x = (x_a, x_b) $$，然后找到条件分布$$ p(x_a|x_b) $$和边缘分布$$ p(x_a) $$的表达式。得到条件分布$$ p(x_a|x_b) $$的均值是关于$$ x_b $$的线性函数。这里，假设给定高斯边缘分布$$ p(x) $$和均值是关于$$ x $$的线性函数且方差与$$ x $$无关的高斯条件分布$$ p(y|x) $$。这是线性高斯模型（linear Gaussian model）的一个例子（Roweis and Ghahramani,
1999），将在8.1.4节学习它更一般的情况。我们希望找到边缘分布$$ p(y) $$和条件分布$$ p(x|y) $$。这个问题在接下来的章节中会经常出现，在这里我们可以方便的证明更一般的结果。    

把边缘和条件分布记为：    

$$
\begin{eqnarray}
p(x) &=& \mathcal{N}(x|\mu,\Lambda^{-1}) \tag{2.99} \\
p(y|x) &=& \mathcal{N}(y|Ax+b, L^{-1}) \tag{2.100}
\end{eqnarray}
$$

其中$$ \mu, A, b $$是控制均值的参数，$$ \Lambda , L $$是精度矩阵。设$$ x,y $$分别是$$ M,D $$维的，那么矩阵$$ A $$是$$ D \times M $$矩阵。    

首先，我们求出$$ x,y $$的联合分布的表达式。为了达到这个目的，我们定义：    

$$
z = 
\left(
\begin{array}{c}  
x \\
y
\end{array}
\right) \tag{2.101}
$$

然后考虑联合分布的对数：

$$
\begin{eqnarray}
\ln p(z) &=& \ln p(x) + \ln p(y|x) \\
&=& -\frac{1}{2}(x - \mu)^T\Lambda(x-\mu) \\
& & -\frac{1}{2}(y-Ax-b)^TL(y-Ax-b) + const \tag{2.102}
\end{eqnarray}
$$

其中$$ const $$表示与$$ x,y $$无关的项。和之前一样，我们得到这是关于$$ z $$的分量的二次函数，因此$$ p(z) $$是高斯分布。为了得到这个分布的精度，考虑式（2.102）中的二次项，它可以写成：     

$$
\begin{eqnarray}
&-&\frac{1}{2}x^T(\Lambda + A^TLA)x - \frac{1}{2}y^TLy + \frac{1}{2}y^TLAx + \frac{1}{2}x^TA^TLy \\
& & = -\frac{1}{2}
\left(
\begin{array}{c}  
x \\
y
\end{array}
\right)^T
\left(
\begin{array}{cc}  
\Lambda + A^TLA & -A^TL\\
-LA & L
\end{array}
\right) 
\left(
\begin{array}{c}  
x \\
y
\end{array}
\right)
=
-\frac{1}{2}z^TRz \tag{2.103}
\end{eqnarray}
$$

得到$$ z $$的高斯分布的精度矩阵（协方差矩阵的逆）为：

$$
R = 
\left(
\begin{array}{cc}  
\Lambda + A^TLA & -A^TL\\
-LA & L
\end{array}
\right) \tag{2.104}
$$

通过矩阵求逆公式（2.76）对精度矩阵求逆可以得到协方差矩阵：    

$$
cov[z] = R^{-1} = 
\left(
\begin{array}{cc}  
\Lambda^{-1} & \Lambda^{-1}A^T  \\
A\Lambda^{-1} & L^{-1} + A\Lambda^{-1}A^T
\end{array}
\right) \tag{2.105}
$$

同样的，我们可以通过确定式（2.102）中的线性项来得到$$ z $$上的高斯分布的均值：    

$$
x^T\Lambda\mu - x^TA^TLb + y^TLb = 
\left( \begin{array}{c} x \\ y \end{array} \right)^T
\left( \begin{array}{c} \Lambda\mu - A^TLb \\ Lb \end{array} \right) \tag{2.106}
$$

使用之前在多元高斯分布中，通过配出平方项得到的的二次项的结果（2.71），可以得到$$ z $$的均值：    

$$
\mathbb{E}[z] = R^{-1}
\left( \begin{array}{c} \Lambda\mu - A^TLb \\ Lb \end{array} \right) \tag{2.107}
$$

使用式（2.105）得到：    

$$
\mathbb{E}[z] = 
\left( \begin{array}{c} \mu \\ A\mu + b \end{array} \right) \tag{2.108}
$$

接下来通过对$$ x $$积分得到边缘分布$$ p(y) $$的表达式。回忆一下，高斯随机向量的分量的边缘分布可以相当简单的使用分区协方差矩阵表示出来。具体来说，它的均值和协方差分别有式（2.92）（2.93）给出。使用式（2.105）和（2.108）可以得到边缘分布$$ p(y) $$的均值和方差：    

$$
\begin{eqnarray}
\mathbb{E}[y] &=& A\mu + b \tag{2.109} \\
cov[y] &=& L^{-1} + A\Lambda^{-1}A^T \tag{2.110}
\end{eqnarray}
$$    

一种特殊情况是$$ A = I $$的时候，这时退化成两个高斯分布的卷积。其中，卷积的均值是两个高斯分布均值的和，卷积的方差的是它们方差的和。    

最后，寻找条件分布$$ p(x|y) $$的表达式。回忆一下，条件分布用分区进度矩阵来表示最简单，如式（2.73）（2.75）。把这些结果代入式（2.105）（2.108）得到条件分布$$ p(x|y) $$的方差和均值为：    

$$
\begin{eqnarray}
\mathbb{E}[x|y] &=& (\Lambda + A^TLA)^{-1}\left\{A^TL(y-b) + \Lambda\mu\right\} \tag{2.111} \\
cov[x|y] &=& (\Lambda + A^TLA)^{-1} \tag{2.112}
\end{eqnarray}
$$    

对这个条件分布的估计可以看成贝叶斯定理的一个例子。可以把分布$$ p(x) $$看成$$ x $$上的先验分布。当观测到$$ y $$后，对应的$$ x $$上的后验分布由条件分布$$ p(x|y) $$来表示。得到边缘分布和条件分布，也可以用$$ p(x|y)p(y) $$来表示联合分布$$ p(z) = p(x)p(y|x) $$。结果总结如下：    

> #### 边缘和条件高斯    
> 对于$$ x $$的边缘高斯分布和$$ y $$关于$$ x $$的条件高斯分布：    
> 
> $$ p(x) = \mathcal{N}(x|\mu,\Lambda^{-1}) \tag{2.113} $$
> 
> $$ p(y|x) = \mathcal{N}(y|Ax +bb,L^{-1}) \tag{2.114} $$
> 
> 那么$$ y $$的边缘分布和$$ x $$关于$$ y $$的条件高斯分布为：
> 
> $$ p(y) = \mathcal{N}(y|A\mu + b,L^{-1} + A\Lambda^{-1}A^T) \tag{2.115} $$
> 
> $$ p(x|y) = \mathcal{N}(x|\Sigma\left\{A^TL(y-b) + \Lambda\mu \right\},\Sigma) \tag{2.116} $$
> 
> 其中
> 
> $$ \Sigma = (\Lambda + A^TLA)^{-1} \tag{2.117} $$

