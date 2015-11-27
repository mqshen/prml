目前为止，我们已经研究了线性动态系统中的推断问题，假设模型的参数$$ \theta = \{A,\Gamma,C,\Sigma,\mu_0,P_0\} $$已知。接下来，我们考虑使用最大似然方法确定这些参数Ghahramani and Hinton，1996b）。由于模型具有潜在变量，因此可以使用第9章讨论的一般形式的EM算法来解决这个问题。    

我们可以按照下面的方法推导线性动态系统的EM算法。让我们将算法在某个特定循环上的模型参数估计值记作$$ \theta^{old} $$。对于这些参数，我们可以运行推断算法来确定潜在变量的后验概率分布$$ p(Z|X,\theta^{old}) $$，或者更精确地确定那些在M步骤中所需的局部后验边缘概率。特别的，我们需要下面的期望    

$$
\begin{eqnarray}
\mathbb{E}[z_m] &=& \hat{\mu}_n \tag{13.105} \\
\mathbb{E}[z_nz_{n-1}^T] &=& \hat{V}_nJ_{n-1}^T + \hat{\mu}_n\hat{\mu}_{n-1}^T \tag{13.106} \\
\mathbb{E}[z_nz_n^T] &=& \hat{V}_n + \hat{\mu}_n\hat{mu}_n^T \tag{13.107}
\end{eqnarray}
$$    

其中我们已经使用了公式（13.104）。    

现在我们考虑完整数据对数似然函数，它通过对式（13.6）取对数的方式得到，因此结果为    

$$
\begin{eqnarray}
\ln p(X,Z|\theta) &=& \ln p(z_1|\mu_0,P_0) + \sum\limits_{n=2}^N\ln p(z_n|z_{n-1},A,\Gamma) \\
& & + \sum\limits_{n=1}^N\ln p(x_n|z_n,C,\Sigma) \tag{13.108}
\end{eqnarray}
$$     

其中我们显式地写出了对参数的依赖关系。我们现在对完整数据对数似然函数关于后验概率分布$$ p(Z|X,\theta^{old}) $$取期望，它定义了函数    

$$
Q(\theta,\theta^{old}) = \mathbb{E}_{Z|\theta^{old}}[\ln p(X,Z|\theta)] \tag{13.109}
$$    

在M步骤中，函数关于$$ \theta $$的分量进行最大化。     

首先考虑参数$$ \mu_0 $$和$$ P_0 $$。如果我们使用（13.77）消去式（13.108）中的$$ p(z_1|\mu_0, P_0) $$，然后关于$$ Z $$取期望，得到    

$$
Q(\theta,\theta^{old}) = -\frac{1}{2}\ln |P_0| - \mathbb{E}_{Z|\theta^{old}}\left[\frac{1}{2}(z_1 - \mu_0)^TP_0^{-1}(z_1 - \mu_0)\right] + const
$$    

其中所有不依赖于$$ \mu_0 $$或$$ P_0 $$的项都被整合到了可加性常数中。使用2.3.4节讨论的高斯分布的最大似然解，关于$$ \mu_0 $$和$$ P_0 $$进行最大化很容易进行，结果为    

$$
\begin{eqnarray}
\mu_0^{new} &=& \mathbb{E}[z_1] \tag{13.110} \\
V_0^{new} &=& \mathbb{E}[z_1z_1^T] - \mathbb{E}[z_1]\mathbb{E}[z_1^T] \tag{13.111}
\end{eqnarray}
$$    

类似的，为了最优化$$ A $$和$$ \Gamma $$，我们使用式（13.75）消去（13.108）中的$$ p(z_n|z_{n−1}, A, \Gamma) $$，结果为    

$$
\begin{eqnarray}
Q(\theta,\theta^{old}) = &-&\frac{N-1}{2}\ln |\Gamma| \\
&-& \mathbb{E}_{Z|\theta^{old}}\left[\frac{1}{2}\sum\limits_{n=2}^N(z_n - Az_{n-1})^T\Gamma^{-1}(z_n - Az_{n-1})\right] + const \tag{13.112}
\end{eqnarray}
$$

其中常数项由不依赖与$$ A $$和$$ \Gamma $$的项组成。关于这些参数最大化可得    

$$
\begin{eqnarray}
A^{new} &=& \left(\sum\limits_{n=2}^N\mathbb{E}[z_nz_{n-1}^T]\right)\left(\sum\limits_{n=2}^N\mathbb{E}[z_{n-1}z_{n-1}^T]\right)^{-1} \tag{13.113} \\
\Gamma^{new} &=& \frac{1}{N-1}\sum\limits_{n=2}^N\Bigg\{\mathbb{E}[z_nz-n^T] - A^{new}\mathbb{E}[z_{n-1}z_n^T] \\
& & -\mathbb{E}[z_nz_{n-1}^T](A^{new})^T + A^{new}\mathbb{E}[z_{n-1}z_{n-1}^T](A^{new})^T\Bigg\} \tag{13.114}
\end{eqnarray}
$$    

注意，$$ A^{new} $$必须首先计算，然后用它的结果来确定$$ \Gamma^{new} $$。    

最后，为了确定$$ C $$和$$ \Sigma $$的新值，我们使用式（13.76）消去式（13.108）中的$$ p(x_n|z_n,C,\Sigma) $$，可得    

$$
\begin{eqnarray}
Q(\theta,\theta^{old}) = &-& \frac{N}{2}\ln |\Sigma| \\
&-&\mathbb{E}_{Z|\theta^{old}}\left[\frac{1}{2}\sum\limits_{n=1}^N(x_n - Cz_n)^T\Sigma^{-1}(x_n - Cz_n)\right] + const
\end{eqnarray}
$$    

关于$$ C $$和$$ \Sigma $$最大化，可得    

$$
\begin{eqnarray}
C^{new} &=& \left(\sum\limits_{n=1}^Nx_n\mathbb{E}[z_n^T]\right)\left(\sum\limits_{n=1}^N\mathbb{E}[z_nz_n^T]\right)^{-1} \tag{13.115} \\
\Sigma^{new} &=& \frac{1}{N}\sum\limits_{n=1}^N\{x_nx_n^T - C^{new}\mathbb{E}[z_n]x_n^T \\
& & -x_n\mathbb{E}[z_n^T](C^{new})^T + C^{new}\mathbb{E}[z_nz_n^T](C^{new})^T\} \tag{13.116}
\end{eqnarray}
$$     

我们得到了使用最大似然方法学习线性动态系统的参数的方法。引入先验概率分布得到MAP估计的方法很简单。使用第10章讨论的近似方法，可以得到一个完整的贝叶斯方法。篇幅所限，不在这里详细介绍这些内容。
