当我们近似分布$$ p(z) $$的时候，也可以获得标准化常数$$ Z $$的近似。使用式（4.133）的近似得到    

$$
\begin{eqnarray}
Z &=& \int f(z)dz \\
&\simeq& f(z_0)\int exp\left\{-\frac{1}{2}(z-z_0)^TA(z-z_0)\right\}dz \\
&=& f(z_0)\frac{(2\pi)^{M/2}}{\vert A \vert^{1/2}} \tag{4.135}
\end{eqnarray}
$$

其中，被积函数是高斯的，且使用了标准高斯分布的标准形式（2.43）。可以使用结果（4.135）来得到3.4节中讨论的在贝叶斯模型比较中起着相当重要的作用的模型证据的近似。    

考虑数据集$$ D $$和具有参数$$ \{\theta_i\} $$的模型集合$$ \{M_i\} $$。对每个模型我们定义一个似然函数$$ p(D|\theta_i,M_i) $$。如果我们引入参数的先验$$ p(\theta_i|M_i) $$，那么我们对计算不同模型的模型证据$$ p(D|M_i) $$比较感兴趣。为了简化记号，从现在开始省略对于$$ M_i $$的条件依赖。根据贝叶斯定理模型证据由    

$$
p(D) = \int p(D|\theta)p(\theta)d\theta \tag{4.136}
$$

令$$ f(\theta) = p(D|\theta)p(\theta) $$且$$ Z = p(D) $$，同时应用结果（4.135），可得

$$
\ln p(D) \simeq \ln p(D|\theta_{MAP}) + \underbrace{\ln p(\theta_{MAP}) + \frac{M}{2}\ln (2\pi) - \frac{1}{2}\ln \vert A \vert}_{Occam factor} \tag{4.137}
$$

其中$$ \theta_{MAP} $$是后验分布的众数的$$ \theta $$值，$$ A $$是负对数后验概率的二阶导数的Hessian矩阵

$$
A = - \nabla\nabla\ln p(D|\theta_{MAP})p(\theta_{MAP}) = -\nabla\nabla\ln p(\theta_{MAP}|D) \tag{4.138}
$$

式（4.137）的右手边的第一项表示使用最优的参数计算得到的对数似然，剩下的三项组成“Occam factor”来惩罚模型的复杂度。    

如果我们假设参数的高斯先验分布比较宽，且Hassian矩阵是满秩的，那么我们可以使用

$$
\ln p(D) \simeq \ln p(D|\theta_{MAP}) - \frac{1}{2}M\ln N \tag{4.139}
$$

来粗略的近似式（4.137）。其中$$ N $$是数据点的个数，$$ M $$是$$ \theta $$中的参数数量，且我们省略的额外的常数。这就是贝叶斯信息准则（Bayesian Information Criterion）（BIC）或Schwarz准则（Schwarz, 1978）。注意，对比（1.73）给出的AIC，它对模型复杂度的惩罚更重。    

像AIC和BIC这样的复杂度的一个优点是很容易计算它们的度量。但也会产生误导性的结果。特别的，由于许多参数都不是“良好确定”的，所以Hessian矩阵满秩的假设通常不成立。我们可以使用基于拉普拉斯近似的式（4.137）来获得对模型证据的一个更加准确的估计，正如我们在5.7节的神经网络模型中做的那样。    


