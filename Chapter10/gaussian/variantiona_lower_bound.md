我们也可以很容易地计算这个模型的下界（10.3）。在实际应用中，能够在重新估计期间监视模型的下界是很有用的，这可以用来检测是否收敛。它也可以为解的数学表达式和它们的软件执行提供一个有价值的检查，因为在迭代重新估计的每个步骤中，这个下界的值应该不会减小。我们可以进一步地使用变分下界检查更新方程的数学推导和它们的软件执行的正确性，方法是使用有限差来检查每次更新确实给出了下界的一个（具有限制条件的）极大值（Svensen and Bishop, 2004）。     

对于高斯分布的变分混合，下界（10.3）为     

$$
\begin{eqnarray}
L &=& \sum\limits_Z\int\int\int q(Z,\pi,\mu,\Lambda)\ln\left\{\frac{p(X,Z,\pi,\mu,\Lambda)}{q(Z,\pi,\mu,\Lambda)}\right\}d\pi d\mu d\Lambda \\
&=& \mathbb{E}[\ln p(X,Z,\pi,\mu,\Lambda)] - \mathbb{E}[\ln q(Z,\pi,\mu,\Lambda)] \\
&=& \mathbb{E}[\ln p(X|Z,\mu,\Lambda)] - \mathbb{E}[\ln q(Z|X)] + \mathbb{E}[\ln p(\pi)] + \mathbb{E}[\ln p(\mu,\Lambda)] \\
& & -\mathbb{E}[\ln q(Z)] - \mathbb{E}[\ln q(\pi)] - \mathbb{E}[\ln q(\mu,\Lambda)] \tag{10.70}
\end{eqnarray}
$$     

其中，为了保持记号简洁，我们省略了$$ q $$分布上的$$ * $$上标，以及期望算符的下标，因为每个期望是关于它的所有参数进行计算的。下界的各项很容易计算，结果为     

$$
\begin{eqnarray}
\mathbb{E}[\ln p(X|Z,\mu,\Lambda)] &=& \frac{1}{2}\sum\limits_{k=1}^KN_k\Bigg\{\ln\tilde{\Lambda}_k - D\beta_k^{-1} - v_k Tr(S_kW_k) \\    
& & - v_k(\bar{x}_k - m_k)^TW_k(\bar{x}_k - m_k) - D\ln(2\pi)\Bigg\} \tag{10.71} \\
\mathbb{E}[\ln p(Z|\pi)] &=& \sum\limits_{n=1}^N\sum\limits_{k=1}^Kr_{nk}\ln\bar{\pi}_k \tag{10.72} \\
\mathbb{E}[\ln p(\pi)] &=& \ln C()\alpha_0) + (\alpha_0 - 1)\sum\limits_{k=1}^K\ln\tilde{\pi}_k \tag{10.73} \\
\mathbb{E}[\ln p(\mu,\Lambda) &=& \frac{1}{2}\sum\limits_{k=1}^K\Bigg\{D\ln\left(\frac{\beta_0}{2\pi}\right) + \ln\tilde{\Lambda}_k - \frac{D\beta_0}{\beta_k} \\ 
& & -\beta_0v_0(m_k - m_0)^TW_k(m_k - m_0)\Bigg\} + K\ln B(W_0,v_0) \\
& & +\frac{v_0 - D - 1}{2}\sum\limits_{k=1}^K\ln\tilde{\Lambda}_k - \frac{1}{2}\sum\limits_{k=1}{K}v_k Tr(W_0^{-1}W_k) \tag{10.74} \\
\mathbb{E}[\ln q(Z)] &=& \sum\limits_{n=1}^N\sum\limits_{k=1}^Kr_{nk}\ln r_{nk} \tag{10.75} \\
\mathbb{E}[\ln q(\pi)] &=& \sum\limits_{k=1}^K(\alpha_k - 1)\ln\tilde{\pi}_k + \ln C(\alpha) \tag{10.76} \\
\mathbb{E}[\ln q(\mu,\Lambda)] &=& \sum\limits_{k=1}^K\left\{\frac{1}{2}\ln\tilde{\lambda}_k + \frac{D}{2}\ln\left(\frac{\beta_k}{2\pi}\right) - \frac{D}{2} - H[q(\Lambda_k)]\right\} \tag{10.77}
\end{eqnarray}
$$

其中$$ D $$是$$ x $$的维度，$$ H[q(\Lambda_k)] $$是式（B.82）给出的Wishart分布的熵，系数$$ C(\alpha) $$和$$ B(W , \nu) $$分别由式（B.23）和式（B.79）定义。注意，涉及到$$ q $$分布的对数的期望的项仅仅表示这些分布的熵的负值。当这些表达式进行加和给出下界的表达式时，某些项可以组合到一起，使表达式得到简化。然而，我们将各个表达式分开写，为了让理解更容易。      

最后，值得注意的一点是，下界提供了另一种推导变分重估计方程的方法（变分重估计方程在10.2.1节已经得到）。为了说明这一点，我们使用下面的事实：由于模型有共轭先验，因此变分后验分布（即$$ Z $$的离散分布、$$ \pi $$的狄利克雷分布以及$$ (\mu_k, \Lambda_k) $$的高斯-Wishart分布）的函数形式是已知的。通过使用这些分布的一般的参数形式，我们可以推导出下界的形式，将下界作为概率分布的参数的函数。关于这些参数最大化下界就会得到所需的重估计方程。 
