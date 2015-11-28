在高斯模型的贝叶斯混合的应用中，我们通常对观测变量的新值$$ x $$的预测概率密度感兴趣。与这个观测相关联的有一个潜在变量$$ \hat{z} $$，从而预测概率分布为     

$$
p(\hat{x}|X) = \sum\limits_{\hat{z}}\int\int\int p(\hat{x}|\hat{z},\mu,\Lambda)p(\hat{z}|\pi)p(\pi,\mu,\Lambda|X)d\pi d\mu d\Lambda \tag{10.78}
$$     

其中$$ p(\pi, \mu, \Lambda | X) $$是参数的（未知）真实后验概率分布。使用式（10.37）和式（10.38），我们可以首先完成在$$ z $$上的求和，得到     

$$
p(\hat{x}|X) = \sum\limits_{k=1}^K\int\int\int\pi_k\mathcal{N}(\hat{x}|\mu_k,\Lambda_k^{-1})p(\pi,\mu,\Lambda|X)d\pi d\mu d\Lambda \tag{10.79}
$$      

由于剩下的积分是无法计算的，因此我们通过将真实后验概率分布$$ p(\pi, \mu, \Lambda | X) $$用它的变分近似$$ q(\pi)q(\mu, \Lambda) $$替换的方式来近似预测概率分布，结果为      

$$
p(\hat{x}|X) \simeq \sum\limits_{k=1}^K\int\int\int\pi_k\mathcal{N}(\hat{x}|\mu_k,\Lambda_k)d\pi d\mu_kd\Lambda_k \tag{10.80}
$$     

其中我们使用了式（10.55）给出的分解方式，且在每一项中，我们已经隐式的将$$ i \neq j$$的全部$$ \{\mu_j,\Lambda_j\} $$变量积分出去。剩余的积分现在可以解析地计算，得到一个学生t分布的混合，即     

$$
p(\hat{x}|X) \simeq \frac{1}{\hat{\alpha}}\sum\limits_{k=1}^K\alpha_kSt(\hat{x}|m_k,L_k,\nu_k + 1 - D) \tag{10.81}
$$     

其中第$$ k $$个分量的均值为$$ m_k $$，精度为      

$$
L_k = \frac{(\nu_k + 1 - D)\beta_k}{1 + \beta_k}W_k \tag{10.82}
$$     

其中$$ \nu_k $$由式（10.63）给出。当数据集的大小$$ N $$很大时，预测分布（10.81）就变成了高斯混合。     

