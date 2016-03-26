我们已经知道，如果一个联合分布$$ p(x_a, x_b) $$是高斯的，那么条件分布$$ p(x_a|x_b) $$也是高斯的。现在我们转而讨论边缘分布：

$$
p(x_a) = \int p(x_a, x_b)dx_b \tag{2.83}
$$

接下来，证明这也是高斯的。同样的，计算这个分布的策略是把注意力集中在联合分布的指数中的二次型上面，然后确定边缘分布$$ p(x_a) $$的均值和方差。    

在式（2.70）中，用分区精度矩阵来表示联合分布的二次型。由于目标是积分掉$$ x_b $$，达到这个目的最简单的方法是：首先为了积分计算的方便，只考虑涉及到$$ x_b $$的项，然后配出平方项。拿出只涉及$$ x_b $$的项：

$$
-\frac{1}{2}x_b^T\Lambda_{bb}x_b + x_b^Tm = -\frac{1}{2}(x_b-\Lambda_{bb}^{-1}m)^T\Lambda_{bb}(x_b-\Lambda_{bb}^{-1}m) + \frac{1}{2}m^T\Lambda_{bb}^{-1}m \tag{2.84}
$$

其中

$$
m = \Lambda_{bb}\mu_b - \Lambda_{ba}(x_a - \mu_a) \tag{2.85}
$$

这样我们就把依赖于$$ x_b $$的项转化为高斯分布的标准二次型（对应式（2.84）右手边第一项）和一个不依赖于$$ x_b $$的项（但是依赖于$$ x_a $$）。取这个二次型为指数项代入式（2.83）得到：

$$
\int exp \left\{-\frac{1}{2}(x_b - \Lambda_{bb}^{-1}m)^T\Lambda_{bb}(x_b - \Lambda_{bb}^{-1}m)\right\}dx_b \tag{2.86}
$$

注意，这是对一个非标准化的高斯分布表达式的积分，所以可以很容易的得到这个积分结果为标准化系数的倒数。从标准高斯分布公式（2.43）可以知道这个系数是不依赖于均值，而只依赖于协方差矩阵的行列式。通过配出$$ x_b $$的平方项，可以积分掉$$ x_b $$，那么式（2.84）左手边对结果有影响的依赖于$$ x_a $$的项只剩下式（2.84）中右手边的最后一项，其中$$ m $$由式（2.85）给出。这项与式（2.70）关于$$ x_a $$的剩余项相结合得到：    

$$
\begin{eqnarray}
&\frac{1}{2}&\left[\Lambda_{bb}\mu_b - \Lambda_{ba}(x_a - \mu_a)\right]^T\Lambda_{bb}^{-1}\left[\Lambda_{bb}\mu_b - \Lambda_{ba}(x_a - \mu_a)\right] \\
& & -\frac{1}{2}x_a^T\Lambda_{aa}x_a + x_a^T(\Lambda_{aa}\mu_a + \Lambda_{ab}\mu_b) + const \\
&=& -\frac{1}{2}x_a^T(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})x_a \\
& & + x_a^T(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1}\mu_a + const \tag{2.87}
\end{eqnarray}
$$

其中$$ const $$是与$$ x_a $$无关的量。再次与式（2.71）比较。得到边缘分布$$ p(x_a) $$的协方差：    

$$
\Sigma_a = (\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1} \tag{2.88}
$$

类似的，均值：

$$
\Sigma_a(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\mu_a = \mu_a \tag{2.89}
$$

其中运用了式（2.88）。式（2.88）是使用式（2.69）给出的分区精度矩阵来表示的。可以和条件分布中做法一样，可以使用式（2.67）对应给出的分区协方差矩阵重写这个式子。这两个矩阵的关系是：    

$$
\left(
\begin{array}{cc}  
\Lambda_{aa} & \Lambda_{ab} \\
\Lambda_{ba} & \Lambda_{bb} \\
\end{array}
\right)^{-1}
=
\left(
\begin{array}{cc}  
\Sigma_{aa} & \Sigma_{ab} \\
\Sigma_{ba} & \Sigma_{bb} \\
\end{array}
\right) \tag{2.90}
$$

代入式（2.76），得到：

$$
\left(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}\right)^{-1} = \Sigma_{aa} \tag{2.91}
$$

这样我们就得到边缘分布$$ p(x_a) $$的符合直觉的的均值及协方差：    
$$
\begin{eqnarray}
\mathbb{E}[x_a] = \mu_a \tag{2.92} \\
cov[x_a] = \Sigma_{aa} \tag{2.93}
\end{eqnarray}
$$

这样就得到与使用分区精度矩阵会得到更加简单的表示形式的条件概率相对的，使用分区协方差矩阵就能很简单的表示均值和协方差的边缘分布。    


分区高斯的边缘或条件分布总结如下：    

> #### 分区高斯    
> 对于联合高斯分布$$ \mathcal{N}(x|\mu,\Sigma) , \Lambda \equiv \Sigma^{-1} $$和
> $$ x = \left( \begin{array}{c}  x_a \\ x_b \end{array} \right), \mu = \left( \begin{array}{c}  \mu_a \\ \mu_b \end{array} \right)  \tag{2.94} $$
> $$ \Sigma = \left( \begin{array}{cc}  \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \\ \end{array} \right), \Lambda = \left( \begin{array}{cc}  \Lambda_{aa} & \Lambda_{ab} \\ \Lambda_{ba} & \Lambda_{bb} \\ \end{array} \right) \tag{2.95} $$
> 
> 条件分布：    
> $$ p(x_a|x_b) = \mathcal{N}(x|\mu_{a|b}, \Lambda_{aa}^{-1}) \tag{2.96} $$
> $$ \mu_{a|b} = \mu_a - \Lambda_{aa}^{-1}\Lambda_{ab}(x_a - \mu_b) \tag{2.97} $$
> 
> 边缘分布：    
> $$ p(x_a) = \mathcal{N}(x_a|\mu_a, \Sigma_{aa}) \tag{2.98} $$

图2.9展示涉及到两个变量的多元高斯分布的条件概率分布和边缘概率分布     

![图 2-9](images/conditional_marginal_gaussian.png)      
图 2.9: 左图给出了两个变量上的高斯分布$$ p(xa, xb) $$的等高线，右图给出了边缘概率分布$$ p(x_a) $$ （蓝色曲线）和$$ x_b = 0.7 $$的条件概率分布$$ p(x_a|x_b) $$（红色曲线）。

