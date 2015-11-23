目前为止，我们将先验概率分布的超参数$$ \alpha $$看成一个已知参数。我们现在将贝叶斯logistic回归模型进行推广，使得这个参数的值可以从数据集中推断出来。这可以通过将全局变分近似和局部变分近似结合到一个框架中的方式完成，从而在每个阶段都保留边缘似然函数的下界。Bishop and Svenen(2003)在研究专家模型的层次混合的贝叶斯方法中，采用了这样一种组合的方法。     

特别的，我们再次考虑一个简单的各向同性的高斯先验概率分布，形式为     

$$
p(w|\alpha) = \mathcal{N}(w|0,\alpha^{-1}I) \tag{10.165}
$$     

我们的分析可以推广到更一般的高斯先验分布中，例如，如果我们希望为参数$$ w_j $$的不同子集关联一个不同的超参数，那么我们就可以将我们的分析进行推广。与之前一样，我们考虑$$ \alpha $$上的共轭超先验，这是一个Gamma分布     

$$
p(\alpha) = Gam(\alpha|\alpha_0,b_0) \tag{10.166}
$$      

它由常数$$ a_0 $$和$$ b_0 $$控制。      

这个模型的边缘似然函数现在的形式为     

$$
p(t) = \int\int p(w,\alpha,t)dwd\alpha \tag{10.167}
$$     

其中，联合概率分布为      

$$
p(w,\alpha,t) = p(t|w)p(w|\alpha)p(\alpha) \tag{10.168}
$$     

我们现在无法直接计算关于$$ w $$和$$ \alpha $$的积分。我们会在同一个模型中使用全局的变分方法和局部的变分方法来解决这个问题。    

首先，我们引入一个变分分布$$ q(w, \alpha) $$，然后应用式（10.2）给出的分解方式。在这种情况     

$$
\ln p(t) = L(q) + KL(q \Vert p) \tag{10.169}
$$

其中，下界$$ L(q) $$和Kullback-Leibler散度$$ KL(q \Vert p) $$的定义为    

$$
\begin{eqnarray}
L(q) &=& \int\int q(w,\alpha)\ln\left\{\frac{p(w,\alpha,t)}{q(w,\alpha)}\right\}dwd\alpha \tag{10.170} \\
KL(q \Vert p) &=& -\int\int q(w,\alpha)\ln\left\{\frac{p(w,\alpha|t)}{q(w,\alpha)}\right\}dwd\alpha \tag{10.171}
\end{eqnarray}
$$    

现在，由于似然因子$$ p(t|w) $$的形式，下界$$ L(q) $$仍然无法求解。于是，与之前一样，我们对每个logistic sigmoid因子应用一个局部的变分界限。这使得我们可以使用不等式（10.152），得到$$ L(q) $$的下界，这个下界也是对数似然函数的一个下界。     

$$
\begin{eqnarray}
\ln p(t) &\geq& L(q) \geq \tilde{L}(q,\xi) \\
&=& \int\int q(w,\alpha)\ln\left\{\frac{h(w,\xi)p(w|\alpha)p(\alpha)}{q(w,\alpha)}\right\}dwd\alpha \tag{10.172}
\end{eqnarray}
$$     

接下来我们假设变分分布可以在参数和超参数之间进行分解，即      

$$
q(w,\alpha) = q(w)q(\alpha)\tag{10.173}
$$     

有了这种分解，我们可以使用式（10.9）给出的一般结果，得到最优因子的表达式。首先考虑概率分布$$ q(w) $$。丢弃与$$ w $$无关的项，我们有     

$$
\begin{eqnarray}
\ln q(w) &=& \mathbb{E}_\alpha[\ln\{h(w,\xi)p(w|\alpha)p(\alpha)\}] + const \\
&=& \ln h(w,\xi) + \mathbb{E}_\alpha[\ln p(w|\alpha)]+ const
\end{eqnarray}
$$      

我们现在使用式（10.153）消去$$ \ln h(w, \xi) $$，使用式（10.165）消去$$ \ln p(w|\alpha) $$，有      

$$
\ln q(w) = -\frac{\mathbb{E}[\alpha]}{2}w^Tw + \sum\limits_{n=1}^N\left\{(t_n - \frac{1}{2})w^T\phi_n - \lambda(\xi_n)w^t\phi_n\phi_n^Tw\right\} + const
$$      

我们看到这是$$ w $$的一个二次函数，因此$$ q(w) $$的解是高斯分布。使用通常的配平方方法，我们有     

$$
q(w) = \mathcal{N}(w|\mu_N,\Sigma_N) \tag{10.174}
$$

其中我们定义了     

$$
\begin{eqnarray}
\Sigma_N^{-1}\mu_N &=& \sum\limits_{n=1}^N\left(t_n - \frac{1}{2}\right)\phi_n \tag{10.175} \\
\Sigma_N^{-1} = \mathbb{E}[\alpha]I + 2\sum\limits_{n=1}^N\lambda(\xi_n)\phi_n\phi_n^T \tag{10.176}
\end{eqnarray}
$$

类似的，因子$$ q(\alpha) $$的最优解为      

$$
\ln q(\alpha) = \mathbb{E}_w[\ln p(w|\alpha)] + \ln p(\alpha) + const 
$$

使用式（10.165）消去$$ \ln p(w|\alpha) $$，使用式（10.166）消去$$ ln p(\alpha) $$，我们有     

$$
\ln q(\alpha) = \frac{M}{2}\ln\alpha - \frac{\alpha}{2}\mathbb{E}[w^Tw] + (\alpha_0 - 1)\ln\alpha - b_0\alpha + const 
$$     

我们看到这是一个Gamma分布的对数，因此我们有     

$$
q(\alpha) = Gam(\alpha|a_N,b_N) = \frac{1}{\Gamma(a_N)}a_N^{b_N}\alpha^{a_N -1}e^{-b_N\alpha} \tag{10.177}
$$    

其中

$$
\begin{eqnarray}
a_N &=& a_0 + \frac{M}{2}\tag{10.178} \\
b_N &=& b_0 + \frac{1}{2}\mathbb{E}_w[w^Tw] \tag{10.179}
\end{eqnarray}
$$

我们还需要最优化变分参数$$ \xi_n $$，这也可以通过最大化下界$$ \tilde{L}(q, \xi) $$的方式得到。略去与$$ \xi $$无关的项，对$$ \alpha $$积分，我们有     

$$
\tilde{L}(q,\xi) = \int q(w)\ln h(w,\xi)dw + const \tag{10.180} 
$$     

注意，它的形式与式（10.160）的形式完全相同，因此我们可以使用我们之前的结果（10.163），它可以通过直接对边缘似然函数的最优化得到，从而重估计方程的形式为     

$$
(\xi^{new})^2 = \phi_n^T(\Sigma_N + \mu_N\mu_N^T)\phi_n \tag{10.181}
$$     

我们已经得到了三个量$$ q(w) , q(\alpha) $$和$$ \xi $$的重估计方程，因此在进行合适的最优化之后，我们可以在这些量之间进行循环，每次都对各个量进行更新。所要求解的各阶矩为     

$$
\begin{eqnarray}
\mathbb{E}[\alpha] = \frac{a_N}{b_N} \tag{10.182} \\
\mathbb{E}[ww^T] = \Sigma_N + \mu_N\mu_N^T \tag{10.183}
\end{eqnarray}
$$

