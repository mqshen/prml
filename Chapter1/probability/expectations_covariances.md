概率中的一个重要操作是找到加权平均值。概率分布$$ p(x) $$的函数$$ f(x) $$的平均值被称为$$ f(x) $$的期望，记作
    
$$ 
\mathbb{E}[f] = \sum\limits_xp(x)f(x) \tag{1.33} 
$$ 

所以平均值是有不同的$$ x $$的概率进行加权的。在连续变量的情形下，期望由对应的概率密度的积分的形式表示:

$$ 
\mathbb{E}[f] = \int p(x)f(x)dx \tag{1.34} 
$$     

两种情形下，如果我们给定有限的$$ N $$个点，这些点满足某个概率分布或概率密度函数，那么期望可以通过求和的方式估计：    

$$ 
\mathbb{E}[f] \simeq \frac{1}{N}\sum\limits_{n=1}^{N}f(x_n) \tag{1.35}
$$    

在第11章中讨论取样方法时，我们会广泛使用这个方法。当$$ N \to \infty $$时，公式(1.35)的估计就精确了。    
有时，我们会考虑多变量函数的期望。这种情形下，我们可以使用下标来表明根据哪个变量进行的平均，例如：    

$$ 
\mathbb{E}_x[f(x, y)] \tag{1.36} 
$$    

表示函数$$ f(x, y) $$关于$$ x $$的分布的平均，注意$$ \mathbb{E}_x[f(x, y)] $$是关于$$ y $$的函数。    

我们同样可以得到关于条件分布的条件期望（conditional expectation）:

$$ 
\mathbb{E}[f|y] ＝ \sum\limits_x p(x|y)f(x)  \tag{1.37} 
$$     

连续变量的定义于此类似。     

$$ f(x) $$方差的定义如下：    

$$ 
var[f] = \mathbb{E}[(f(x) - \mathbb{E}[f(x)])^2] \tag{1.38} 
$$    

它度量了$$ f(x) $$与均值$$ \mathbb{E}[f(x)] $$之间的变异性的程度。把平方展开，方差可以写成$$ f(x) $$与$$ f(x)^2 $$的期望的形式：    

$$
var[f] = \mathbb{E}[f(x)^2] − \mathbb{E}[f(x)]^2 \tag{1.39}
$$

特别的，变量$$ x $$自身的方差可以表示为：    

$$
var[x] = \mathbb{E}[x^2] − \mathbb{E}[x]^2 \tag{1.40}
$$

对于两个变量$$ x, y$$，他们的协方差（corvariance）定义为：     

$$
\begin{eqnarray}
cov[x, y] &=& \mathbb{E}_{x,y}[\{x − \mathbb{E}[x]\} \{y − \mathbb{E}[y]\}]  \\
&=& \mathbb{E}_{x,y}[xy] − \mathbb{E}[x]\mathbb{E}[y] \tag{1.41}
\end{eqnarray}
$$  

它表示$$ x, y $$在多大程度上协同变化。如果$$ x, y $$相互独立，那么它们之间的协方差为0。      

如果$$ x, y $$是两个随机变量的向量，那么他们的协方差是一个矩阵。

$$
\begin{eqnarray}
cov[x, y] &=& \mathbb{E}_{x,y}[\{x − \mathbb{E}[x]\} \{y^T − \mathbb{E}[y^T]\}]  \\
&=& \mathbb{E}_{x,y}[xy^T] − \mathbb{E}[x]\mathbb{E}[y^T] \tag{1.42}
\end{eqnarray}
$$  

如果我们考虑向量$$ x $$各分量之间的协方差，可以稍微简化下我们的记法：$$ cov[x] \equiv cov[x, x] $$
