给定一个新的输入$$ x $$，使用参数的高斯变分后验概率很容易计算出$$ t $$上的预测分布，即     

$$
\begin{eqnarray}
p(t|x,t) &=& \int p(t|x,W)p(w|t)dw \\
&\simeq& \int p(t|x,w)q(w)dw \\
&=& \int\mathcal{N}(t|w^T\phi(x),\beta^{-1})\mathcal{N}(w|m_N,S_N)dw \\
&=& \mathcal{N}(t|m_N^T\phi(x),\sigma^2(x)) \tag{10.105}
\end{eqnarray}
$$    

其中我们使用了式（2.115）给出的线性高斯模型的结果计算积分。这里，与输入相关的方差为    

$$
\sigma^2(x) = \frac{1}{\beta} + \phi(x)^TS_N\phi(x) \tag{10.106}
$$    

注意，这与我们固定$$ \alpha $$得到的结果（3.59）的形式相同，唯一的区别在于现在期望值$$ \mathbb{E}[\alpha] $$出现在$$ S_N $$的定义中。  

