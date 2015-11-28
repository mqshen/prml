我们的第一个目标是寻找对后验概率分布$$ p(w, \alpha|t) $$的一个近似。为了完成这件事，我们使用10.1节的变分框架，变分后验概率分布的分解表达式为     

$$
q(w,\alpha) = q(w)q(\alpha) \tag{10.91}
$$     

我们可以使用式（10.9）给出的一般结果来找到这个分布中的因子的重估计方程。回忆一下，对于每个因子，我们取所有变量上的联合概率分布的对数，然后关于不在这个因子中的变量求平均。首先考虑$$ \alpha $$上的概率分布。只保留与$$ \alpha $$有函数依赖关系的项，我们有     

$$
\begin{eqnarray}
\ln q^*(\alpha) &=& \ln p(\alpha) + \mathbb{E}_w[\ln p(w|\alpha)] + const \\
&=& (a_0 - 1)\ln\alpha - b_0\alpha + \frac{M}{2}\ln\alpha - \frac{\alpha}{2}\mathbb{E}[w^Tw] + const \tag{10.92}
\end{eqnarray}
$$     

我们看到，这是Gamma分布的对数，因此通过观察$$ \alpha $$和$$ \ln \alpha $$的系数，我们有     

$$
q^*(\alpha) = Gam(\alpha|a_N,b_N) \tag{10.93}
$$     

其中    

$$
\begin{eqnarray}
a_N = a_0 + \frac{M}{2} \tag{10.94} \\
b_N = b_0 + \frac{1}{2}\mathbb{E}[w^Tw] \tag{10.95}
\end{eqnarray}
$$

类似的，我们可以找到$$ w $$上的后验概率分布的变分重估计方程。同样的，使用一般的结果（10.9），只保留与$$ w $$有函数依赖关系的项，得到    

$$
\begin{eqnarray}
\ln q^*(w) &=& \ln p(t|w) + \mathbb{E}_\alpha[\ln p(w|\alpha)] + const \tag{10.96} \\
&=& -\frac{\beta}{2}\sum\limits_{n=1}^N\{w^T\phi_n - t_n\}^2 - \frac{1}{2}\mathbb{E}[\alpha]w^Tw + const \tag{10.97} \\
&=& -\frac{1}{2}w^T(\mathbb{E}[\alpha]I + \beta\Phi^T\Phi)w + \beta w^T\Phi^Tt + const \tag{10.98}
\end{eqnarray}
$$     

由于这是一个二次型，因此分布$$ q^*(w) $$是一个高斯分布，因此我们可以使用一般的配平方的方法，得到均值和协方差，结果为     

$$
q^*(w) = \mathcal{N}(w|m_N,S_N) \tag{10.99}
$$     

其中

$$
\begin{eqnarray}
m_N &=& \beta S_N\Phi^Tt \tag{10.100} \\
S_N &=& (\mathbb{E}[\alpha]I + \beta\Phi^T\Phi)^{-1} \tag{10.101}
\end{eqnarray}
$$    

注意这个结果与$$ \alpha $$被当成固定参数时得到的后验概率分布（3.52）的相似性。区别在于，这里$$ \alpha $$被替换为了它在变分分布下的期望$$ \mathbb{E}[\alpha] $$。实际上，在两种情形中，我们选择使用了同样的协方差矩阵$$ S_N $$的记号。    

使用标准结果（B.27）、（B.38）和（B.39），我们可以得到所需的矩，形式为     

$$
\begin{eqnarray}
\mathbb{E}[\alpha] &=& \frac{a_N}{b_N} \tag{10.102} \\
\mathbb{E}[ww^T] &=& m_Nm_N^T + S_N \tag{10.103}
\end{eqnarray}
$$    

变分后验概率分布的计算在开始时，对$$ q(w) $$或$$ q(\alpha) $$中的一个概率分布的参数进行初始化，然后交替地重新更新这些因子，直到满足一个合适的收敛准则（通常根据下界来确定，稍后讨论）。    

将变分方法得到的解与3.5节使用模型证据得到的解练习起来是很有意义的。考虑$$ a_0 = b_0 = 0 $$的情形，对应于$$ \alpha $$上的一个无限宽的鲜艳概率分布。变分后验概率$$ q(\alpha) $$的均值为     

$$
\mathbb{E}[\alpha] = \frac{a_N}{b_N} = \frac{M / 2}{\mathbb{E}[w^Tw] / 2} = \frac{M}{m_N^Tm_N + Tr(S_N)} \tag{10.104}
$$      

与式（9.63）进行对比，表明在这种特别简单的模型中，变分方法得到的解与使用EM算法最大化模型证据函数的方法得到的解完全相同，唯一的区别是$$ \alpha $$的点估计被替换为了它的期望 值。由于分布$$ q(w) $$只通过期望$$ \mathbb{E}[\alpha] $$对$$ q(\alpha) $$产生依赖，因此我们看到这两种方法对于无限宽的先验概率分布会给出相同的结果。
