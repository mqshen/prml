给定一个新特征向量$$ \phi(x) $$，类$$ C_1 $$的预测分布可以通过边缘化由高斯分布$$ q(w) $$近似得到的后验分布$$ p(w|t) $$来获得：    

$$
p(C_1|\phi, t) = \int p(C_1|\phi,w)p(w|t)dw \simeq \int \sigma(w^T\phi)q(w)dw \tag{4.145}
$$

且类别$$ C_2 $$的对应的概率为$$ p(C_2|\phi, t) =  1 - p(C_1|\phi, t) $$。为了计算预测分布，注意，函数$$ \sigma(w^T\phi) $$只依赖于$$ w $$在$$ \phi $$的投影。记$$ a = w^T\phi $$得到     

$$
\sigma(w^T\phi) = \int \delta(a - w^T\phi)\sigma(a)da \tag{4.146}
$$

其中$$ \delta(\dot) $$是Dirac delta函数。根据这些得到

$$
\int \sigma(w^T\phi)q(w)dw = \int\delta(a)p(a)da \tag{4.147}
$$

其中

$$
p(a) = \int\delta(a-w^T\phi)q(w)dw \tag{4.148}
$$

注意，Delta函数对$$ w $$施加了一个线性限制，因此在所有与$$ \phi $$正交的方向上积分，就得到了联合概率分布$$ q(w) $$的边缘分布，这样就得到$$ p(a) $$。因为$$ q(w) $$是高斯的，从2.3.2节中我们知道边缘分布也是高斯的。我们可以通过动差然后交换$$ a, w $$的积分顺序来计算这个分布的均值和方差，即：

$$
\mu_a = \mathbb{E}[a] = \int p(a)ada = \int q(w)w^T\phi dw = w_{MAP}^T\phi \tag{4.149}
$$

其中我们使用了式（4.144）给出的后验分布$$ q(w) $$的结果。类似的

$$
\begin{eqnarray}
\sigma_a^2 &=& var[a] = \int p(a)\{a^2 - \mathbb{E}[a]^2\}da \\
&=& \int q(w)\{(w^T\phi)^2 - (m_N^T\phi)^2\}dw = \phi^TS_N\phi \tag{4.150}
\end{eqnarray}
$$

注意，令噪声方差为零，$$ a $$的分布的函数形式与线性回归模型的预测分布（3.58）相同。因此对预测分布的近似变成了

$$
p(C_1|t) = \int \sigma(a)p(a)da = \int\sigma(a)\mathcal{N}(a|\mu_a,\sigma_a^2)da \tag{4.151}
$$

这个结果也可以直接通过2.3.2节给出的高斯分布的边缘概率的结果推导出来。    

对$$ a $$的积分表示高斯分布和logistic sigmoid函数的卷积，不能解析的计算。然而，可以通过（4.59）定义的logistic sigmoid函数$$ \sigma(a) $$和（4.114）定义的probit函数$$ \Phi(a) $$间的相似性来得到一个较好的近似（Spiegelhalter and Lauritzen, 1990; MacKay, 1992b; Barber and Bishop, 1998a）。为了得到logistic函数最好的近似，需要重新定义横轴标度来使$$ \Phi(\lambda a) $$近似$$ \sigma(a) $$。令两个函数在原点处有同样的斜率，可以找到$$ \lambda $$的合适值：$$ \lambda^2 = \pi / 8 $$。这个$$ \lambda
$$的选择的logistic sigmoid函数和probit函数在图4.9中展示。    

使用probit函数的一个优势是它与高斯的卷积可以用另一个probit函数解析地表示出来。特别的，我们可以证明

$$
\int\Phi(\lambda a)\mathcal{N}(a|\mu,\sigma^2)da = \Phi\left(\frac{\mu}{(\lambda^{-2}+\sigma^2)^{1/2}}\right) \tag{4.152}
$$

现在，对方程的两边应用probit函数的近似$$ \sigma(a) \simeq \Phi(\lambda a) $$，得到logistic sigmoid函数与高斯的卷积近似：

$$
\int \sigma(a)\mathcal{N}(a|\mu,\sigma^2)da \simeq \sigma(\kappa(\sigma^2)\mu) \tag{4.153}
$$

其中我们定义

$$
\kappa(\sigma^2) = (1+\pi\sigma^2/8)^{-1/2} \tag{4.154}
$$

对式（4.151）应用这个结果得到预测分布的近似形式：

$$
p(C_1|\phi,t) = \sigma\left(\kappa(\sigma_a^2)\mu_a\right) \tag{4.155}
$$

其中$$ \mu_a, \sigma_a^2 $$分别由（4.149）（4.150）定义，$$ \kappa(\sigma^2) $$由（4.154）定义。     

注意，对应$$ p(C_1|\phi, t) = 0.5 $$的决策边界由$$ \mu_a = 0 $$给出，这与使用MAP得到的$$ w $$的值得到的决策边界相同。因此，如果决策准则是基于先验概率相同的最小分类错误率，那么对$$ w $$的边缘化是没有效果的。然而，对于更复杂的决策准则，它就起着重要的作用。在后验概率分布的高斯近似下，对logistic sigmoid模型的边缘化将在变量推断的问题下的图10.13中进行说明。    


