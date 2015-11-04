具有高斯噪声分布的线性回归模型，误差函数，对应的负对数似然函数，由式（3.12）给出。如果我们在数据点$$ n $$处，关于对误差函数贡献的参数向量$$ w $$求导数，它具有“误差”$$ y_n - t_n $$乘以特征向量$$ \phi_n $$的形式，其中$$ y_n = w^T\phi_n $$。同样的，对于结合logistic
sigmoid激活函数和交叉熵误差函数（4.90），或多类别交叉熵误差函数的softmax激活函数（4.108），我们都获得了同样简单的形式。现在我们证明，如果假设目标变量的条件分布是指数族分布，对应的激活函数为标准链接函数（canonical link function），那么这个结果是一个一般的结果。    

再次使用式（4.84）给出的指数族分布限制。注意，这里我们把指数族分布的假设应用于目标变量$$ t $$，而不是4.2.4节中应用于输入向量$$ x $$。考虑目标变量的条件分布形式

$$
p(t|\eta,s) = \frac{1}{s}h\left(\frac{h}{s}\right)g(\eta)exp\left\{\frac{\eta t}{s}\right\} \tag{4.118}
$$

使用与推导式（2.226）时相同的方法，得到$$ t $$的条件均值（记作$$ y $$）为

$$
y \equiv \mathbb{E}[t|\eta] = -s\frac{d}{d\eta}\ln g(\eta) \tag{4.119}
$$

因此$$ y, \eta $$一定相关，我们把这个关系记作$$ \eta = \varphi(y) $$。    

根据Nelder and Wedderburn(1972)的方法，我们把广义线性模型定义成$$ y $$是输入（或特征）变量线性组合的非线性函数，即

$$
y = f(w^T\phi) \tag{4.120}
$$

其中$$ f(\dot) $$在机器学习中被称为激活函数，$$ f^{-1}(\dot) $$在统计学中被称为连接函数。    

现在考虑这个模型的对数似然函数，这是一个关于$$ \eta $$的函数，由

$$
\ln p(t|\eta,s) = \sum\limits_{n=1}^N \ln p(t_n|\eta,s) = \sum\limits_{n=1}^N \left\{\ln g(\eta_n) + \frac{\eta_nt_n}{s}\right\} + const \tag{4.121}
$$

其中我们假设所有观测共享共同的缩放参数（对应如服从高斯分布噪声的方差），因此$$ s, n $$是无关的。关于模型参数$$ w $$的对数似然函数的导数为

$$
\begin{eqnarray}
\nabla_w\ln p(t|\eta,s) &=& \sum\limits_{n=1}^N\left\{\frac{d}{d\eta_n}\ln g(\eta_n) + \frac{t_n}{s}\right\}\frac{d\eta_n}{dy_n}\frac{dy_n}{da_n}\nabla a_a \\
&=& \sum\limits_{n=1}^N\frac{1}{s}\{t_n - y_n\}\varphi'(y_n)f'(a_n)\phi_n \tag{4.122}
\end{eqnarray}
$$

其中$$ a_n = w^T\phi_n $$，且一起使用了$$ y_n = f(a_n) $$和式（4.119）关于$$ \mathbb{E}[t|\eta] $$的结果。现在，我们看到，如果我们的链接函数$$ f^{-1}(y) $$的形式为

$$
f^{-1}(y) = \varphi(y) \tag{4.123}
$$

那么表达式会得到极大的简化。这得到$$ f(\varphi(y)) = y $$，因此$$ f'(\varphi)\varphi'(y) = 1 $$。同样的，由于$$ a = f^{-1}(y) $$，得到$$ a = \varphi $$，因此$$ f'(a)\varphi'(y) = 1 $$。这种情况下，误差函数的梯度退化为    

$$
\nabla\ln E(w) = \frac{1}{s}\sum\limits_{n=1}^N\{y_n - t_n\}\phi_n \tag{4.124}
$$

对于高斯分布$$ s = \beta^{−1} $$，而对于logistic模型$$ s = 1 $$。

