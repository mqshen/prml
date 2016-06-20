一旦我们有了类条件密度$$ p(x|C_k) $$的参数函数形式后，就可以和先验类概率$$ p(C_k) $$一起使用最大似然来确定参数的值。这就需要观测值$$ x $$和对应的类标签一起组成的数据集。    

首先考虑二分类的情形，每个都是共享协方差矩阵的高斯类条件密度，并假设我们有数据集$$ \{x_n, t_n\}, n = 1,...,N $$。这里$$ t_n = 1 $$标识类$$ C_1 $$，$$ t_n = 0 $$标识类$$ C_2 $$。把先验类概率记作$$ p(C_1) = \pi $$，所以$$ p(C_2) = 1 - \pi $$。对于一个类$$ C_1 $$的点，我们有$$ t_n = 1 $$且

$$
p(x_n, C_1) = p(C_1)p(x_n|C_1) = \pi\mathcal{N}(x_n|\mu_1,\Sigma)
$$

同样的对于类$$ C_2 $$，我们有$$ t_n = 0 $$且

$$
p(x_n, C_2) = p(C_2)p(x_n|C_2) = (1 - \pi)\mathcal{N}(x_n|\mu_2,\Sigma)
$$

因此似然函数由

$$
p(\textbf{t},X|\pi,\mu_1,\mu_2,\Sigma) = \prod\limits_{n=1}^N[\pi\mathcal{N}(x_n|\mu_1,\Sigma)]^{t_n}[(1- \pi)\mathcal{N}(x_n|\mu_2,\Sigma)]^{1 - t_n} \tag{4.71}
$$

其中$$ \textbf{t} = (t_1,...,t_N)^T $$。通常最大化似然函数的对数比较方便。首先考虑关于$$ \pi $$来最大化。对数似然函数中依赖$$ \pi $$的项是

$$
\sum\limits_{n=1}^N{t_n\ln\pi + (1-t_n)\ln(1-pi)} \tag{4.72}
$$

关于$$ \pi $$求导并使其等于0，整理可得

$$
\pi = \frac{1}{N}\sum\limits_{n=1}^Nt_n = \frac{N_1}{N} = \frac{N_1}{N_1 + N_2} \tag{4.73}
$$

其中$$ N_1 $$表示类$$ C_1 $$数据点的总数，$$ N_2 $$表示类$$ C_2 $$数据点的总数。因此$$ \pi $$的最大似然估计和预期的一样是类别$$ C_1 $$的点所占的比例。把先验概率的最大似然估计关联为类别$$ C_k $$的数据点数量占训练集总 数据的比例，就很容易的把它推广到多分类的情况。    

现在，考虑关于$$ \mu_1 $$的最大化。同样的选择对数似然函数中依赖$$ \mu_1 $$的项：

$$
\sum\limits_{n=1}^Nt_n\ln \mathcal{N}(x_n|\mu_1,\Sigma) = -\frac{1}{2}\sum\limits_{n=1}^Nt_n(x_n-\mu_1)^T\Sigma^{-1}(x_n-\mu_1) + const \tag{4.74}
$$

关于$$ \mu_1 $$求导并使其等于0，整理可得

$$
\mu_1 = \frac{1}{N_1}\sum\limits_{n=1}^Nt_nx_n \tag{4.75}
$$

这就是所有类别为$$ C_1 $$的输入向量$$ x_n $$的均值。通过简单的整理，$$ \mu_2 $$对应的解为

$$
\mu_2 = \frac{1}{N_2}\sum\limits_{n=1}^N(1-t_n)x_n \tag{4.76}
$$

同样的，这就是所有类别为$$ C_2 $$的输入向量$$ x_n $$的均值。    

最后，考虑共享的协方差矩阵$$ \Sigma $$的最大似然解。选出对数似然函数中依赖$$ \Sigma $$的项，得到

$$
\begin{eqnarray}
&-&\frac{1}{2}\sum\limits_{n=1}^Nt_n\ln \vert \Sigma \vert - \frac{1}{2}\sum\limits_{n=1}^Nt_n(x_n-\mu_1)^T\Sigma^{-1}(x_n-\mu_1) \\
&-&\frac{1}{2}\sum\limits_{n=1}^N(1-t_n)\ln\vert \Sigma \vert - \frac{1}{2}\sum\limits_{n=1}^N(1-t_n)(x_n-\mu_2)^T\Sigma^{-1}(x_n-\mu_2) \\
&=& -\frac{N}{2}\ln\vert \Sigma \vert - \frac{N}{2} Tr\left\{\Sigma^{-1}S\right\} \tag{4.77}
\end{eqnarray}
$$

其中，我们定义了

$$
\begin{eqnarray}
S &=& \frac{N_1}{N}S_1 + \frac{N_2}{N}S_2 \tag{4.78} \\
S_1 &=& \frac{1}{N_1}\sum\limits_{n \in C_1}(x_n - \mu_1)(x_n - \mu_1)^T \tag{4.79} \\
S_2 &=& \frac{1}{N_2}\sum\limits_{n \in C_2}(x_n - \mu_2)(x_n - \mu_2)^T \tag{4.80}
\end{eqnarray}
$$

使用高斯分布的最大似然解的标准结果，得到$$ \Sigma = S $$，这表示与两个类别都有关系的协方差矩阵的加权平均。    

这个结果很容易推广到$$ K $$个分类问题，来通过最大似然方法，求解每个类别的条件密度都是高斯分布，且协方差矩阵相同的对应参数。注意，因为高斯最大似然估计没有健壮性，类的高斯分布的拟合方法对于离群点并不健壮。    


