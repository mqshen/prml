让我们先考虑关于$$ \alpha $$来最大化$$ p(\textbf{t}|\alpha,\beta) $$。首先定义特征方程：    

$$
(\beta\Phi^T\Phi)u_i = \lambda_iu_i \tag{3.87}
$$

根据式（3.81）得到$$ A $$的特征值$$ \alpha + \lambda_i $$。现在考虑式（3.86）中涉及$$ \ln \vert A \vert $$的项关于$$ \alpha $$的导数。得到：     

$$ 
\frac{d}{d\alpha}\ln\vert A \vert = \frac{d}{d\alpha}\ln\prod\limits_i(\lambda_i + \alpha) = \frac{d}{d\alpha}\sum\limits_i\ln(\lambda_i + \alpha) = \sum\limits_i\frac{1}{\lambda_i + \alpha} \tag{3.88}
$$

因此式（3.86）关于$$ \alpha $$的驻点满足

$$
0 = \frac{M}{2\alpha} - \frac{1}{2}m_N^Tm_N - \frac{1}{2}\sum\limits_i\frac{1}{\lambda_i + \alpha} \tag{3.89}
$$

乘以$$ 2\alpha $$，整理，可得

$$
\alpha m_N^Tm_N = M - \alpha\sum\limits_i\frac{1}{\lambda_i+\alpha} = \gamma \tag{3.90}
$$

由于关于$$ i $$的求和项有$$ M $$个，因此$$ \gamma $$可以写成：    

$$
\gamma = \sum\limits_i\frac{\lambda_i}{\alpha + \lambda_i} \tag{3.91}
$$

稍后讨论$$ \gamma $$的解释。根据方程（3.90），得到使得边缘似然函数最大化的$$ \alpha $$满足：    

$$
\alpha = \frac{\gamma}{m_N^Tm_N} \tag{3.92}
$$

注意，这是$$ alpha $$的一个隐式解，不仅仅因为$$ \gamma $$与$$ \alpha $$相关，还因为后验分布众数$$ m_N $$本身也与$$ \alpha $$的选择有关。因此我们使用迭代的方法求解。首先我们选择一个$$ \alpha $$的初始值，再把这个初始值代入式（3.53）求得$$ m_N $$，利用式（3.91）计算得到$$ \gamma $$，然后这些值再被代入式（3.92）来重新估计$$ \alpha $$的值。 不断执行这个过程直至收敛。注意，由于矩阵$$ \Phi^T\Phi
$$是固定的，因此我们只需要在最开始的时候计算一次特征值，然后接下来只需乘以$$ beta $$就可以得到$$ \lambda_i $$的值。    

需要强调的是，$$ \alpha $$的值是完全通过训练集的数据来确定的。最大似然方法不同，模型的最优的复杂度不需要独立的数据集。    

类似地，我们可以关于$$ \beta $$最大化对数边缘似然函数（3.86）。为了达到这个目的，我们注意到由式（3.87）定义的特征值$$ \lambda_i $$正比于$$ \beta $$，即$$ d\lambda_i/d\beta = \lambda_i/\beta $$，得到：   

$$
\frac{d}{d\beta}\ln\vert A\vert = \frac{d}{d\beta}\sum\limits_i\ln(\lambda_i + \alpha) = \frac{1}{\beta}\sum\limits_i\frac{\lambda_i}{\lambda_i+\alpha} = \frac{\gamma}{\beta} \tag{3.93}
$$

因此，边缘似然的驻点满足：    

$$
0 = \frac{N}{2\beta}-\frac{1}{2}\sum\limits_{n=1}^N\{t_n - m_N^T\phi(x_n)\}^2 - \frac{\gamma}{2\beta} \tag{3.94}
$$

整理可得：    

$$
\frac{1}{\beta} = \frac{1}{N - \gamma}\sum\limits_{n=1}^N\{t_n - m_N^T\phi(x_n)\}^2 \tag{3.95}
$$

与之前一样，这是$$ \beta $$的一个隐式解，可以通过迭代的方法解出。首先选择$$ \beta $$的一个初始值，然后使用这个初始值计算$$ m_N, \gamma $$，然后使用式（3.95）重新估计$$ \beta $$的值，重复直到收敛。如果$$ \alpha, \beta $$的值都是由数据确定的，那么他们的值可以在每次更新$$ \gamma $$之后一起重新估计。
