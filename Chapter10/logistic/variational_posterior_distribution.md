这里，我们会使用一种基于10.5节介绍的局部界限的变分方法。这使得logistic回归的似然函数(由logistic sigmoid函数控制)可以有指数的二次形式近似。因此，与之前一样，比较方便的做法是选择形式为(4.140)的共轭高斯先验。现阶段，我们会将超参数$$ m_0 $$和$$ S_0 $$看成固定的常数。在10.6.3节，我们会展示变分形式如何扩展到超参数未知的情形，这种情况下，超参数的值要从数据中进行推断。      

在变分的框架上，我们寻找边缘似然函数的下界的最大值。对于贝叶斯logistic回归模型，边缘似然函数的形式为     

$$
p(t) = \int p(t|w)p(w)dw = \int\left[\prod\limits_{n=1}^Np(t_n|w)\right]p(w)dw \tag{10.147}
$$     

首先，我们注意到$$ t $$的条件概率分布可以写成     

$$
\begin{eqnarray}
p(t|w) &=& \sigma(a)^t\{1 - \sigma(a)\}^{1-t} \\
&=& \left(\frac{1}{1 + e^{-a}}\right)^t\left(1 - \frac{1}{1 + e^{-a}}\right)^{1-t} \\
&=& e^{at}\frac{e^{-a}}{1+e^{-a}} = e^{at}\sigma(-a) \tag{10.148}
\end{eqnarray}
$$    

其中$$ a = w^T\phi $$。为了得到$$ p(t) $$的下界，我们使用式（10.144）给出的logistic sigmoid函数的变分下界。为了方便，我们在这里重新写一下。    

$$
\sigma(z) \geq \sigma(\xi)exp\left\{\frac{z-\xi}{2} - \lambda(\xi)(z^2 - \xi^2)\right\} \tag{10.149}
$$     

其中     

$$
\lambda(\xi) = \frac{1}{2\xi}\left[\sigma(\xi) - \frac{1}{2}\right] \tag{10.150}
$$     

于是，得到

$$
p(t|w) = e^{at}\sigma(-a) \geq e^{at}\sigma(\xi)exp\left\{-\frac{a+\xi}{2} - \lambda(\xi)(a^2 - \xi^2)\right\} \tag{10.151}
$$    

注意，由于这个下界分别作用于似然函数的每一项，因此存在一个变分参数$$ \xi_n $$，对应于训练集的每个观测$$ (\phi_n, t_n) $$。使用$$ a = w^T\phi $$，乘以先验概率分布，我们可以得到下面的$$ t $$和$$ w $$的联合概率分布。    

$$
p(t,w) = p(t|w)p(w) \geq h(w,\xi)p(w) \tag{10.152}
$$     

其中，$$ \xi $$表示变分参数的集合$$ \{\xi_n\} $$，并且     

$$
\begin{eqnarray}
h(w,\xi) &=& \prod\limits_{n=1}^N\sigma(\xi_n)exp\{w^T\phi_nt_n - (w^T\phi_n + \xi_n) / 2 \\
& & -\lambda(\xi_n)([w^T\phi_n]^2 - \xi_n^2)\} \tag{10.153}
\end{eqnarray}
$$     

精确计算这个后验概率分布需要对不等式的左侧进行标准化。由于这是无法计算的，因此我们反过来对右侧进行操作。注意，右侧的函数不能看成一个概率密度，因为它没有被标准化。但是，一旦它被标准化，来表示一个后验概率分布$$ q(w) $$，它就不再表示下界了。      

由于对数函数是单调递增的函数，因此不等式$$ A \geq B $$表示$$ \ln A \geq \ln B $$。这给出了$$ t $$和$$ w $$之间的联合概率分布的对数的下界，形式为    

$$
\begin{eqnarray}
\ln\{p(t|w)p(w)\} \geq & & \ln p(w) + \sum\limits_{n=1}^N\{\ln\sigma(\xi_n) + w^T\phi_nt_n \\
& & -(w^T\phi_n + \xi_n)/2 - \lambda(\xi_n)([w^T\phi_n]^2 - \xi_n^2(\} \tag{10.154}
\end{eqnarray}
$$     

代入先验概率分布$$ p(w) $$，不等式的右侧变成了一个关于$$ w $$的函数，形式为     

$$
\begin{eqnarray}
-\frac{1}{2}(w - m_0)^TS_0^{-1}(w - m_0) \\
+ \sum\limits_{n=1}^N\{w^T\phi_n(t_n - 1/2) - \lambda(\xi_n)w^T(\phi_n\phi_n^T)w\} + const \tag{10.155}
\end{eqnarray}
$$

这是$$ w $$的一个二次函数，因此我们可以通过分裂出$$ w $$的线性项和二次项，得到后验概率分布的对应的变分近似，这是一个高斯变分后验概率，形式为     

$$
q(w) = \mathcal{N}(w|m_N,S_N) \tag{10.156}
$$    

其中    

$$
\begin{eqnarray}
m_N = S_N\left(S_0^{-1}m_0 + \sum\limits_{n=1}^N\left(t_n + \frac{1}{2}\right)\phi_n\right) \tag{10.157} \\
S_N^{-1} = S_0^{-1} + 2\sum\limits_{n=1}^N\lambda(\xi_n)\phi_n\phi_n^T \tag{10.158}
\end{eqnarray}
$$    

与拉格朗日框架一样，我们又一次得到了对后验概率分布的一个高斯近似。然后，变分参数$$ \{\xi_n\} $$提供的额外的灵活性使得这个近似的精度更高（Jaakkola and Jordan, 2000）。    

这里，我们考虑了一个批量学习的问题，其中所有的训练数据能够一次全部得到。然而，贝叶斯方法本质上相当适用于顺序学习的问题，其中数据点每次只处理一个，然后被丢弃。得到顺序学习情形下的变分方法的公式是很容易的。     

注意，式（10.149）给出的下界只适用于二分类问题，因此这个方法不能直接推广到$$ K > 2 $$个类别的多类问题。Gibbs(1997)研究了多分类问题的另一种下界的形式。    

