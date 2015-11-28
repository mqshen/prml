另一个很重要的量是下界L，定义为     

$$
\begin{eqnarray}
L(q) &=& \mathbb{E}[\ln p(w,\alpha,t)] - \mathbb{E}[\ln q(w,\alpha)] \\
&=& \mathbb{E}_w[\ln p(t|w)] + \mathbb{E}_{w,\alpha}[\ln p(w|\alpha)] + \mathbb{E}_\alpha[\ln p(\alpha)] \\
& & - \mathbb{E}_\alpha[\ln q(w)]_w - \mathbb{E}[\ln q(\alpha)] \tag{10.107}
\end{eqnarray}
$$     

使用之前章节得到的结果，计算各项的值是很容易的，结果为     

$$
\begin{eqnarray}
\mathbb{E}[\ln p(t|w)]_w &=& \frac{N}{2}\ln\left(\frac{\beta}{2\pi}\right) - \frac{\beta}{2}t^Tt + \beta m_N^T\Phi^Tt \\
& & -\frac{\beta}{2}Tr[\Phi^T\Phi(m_Nm_N^T + S_N)] \tag{10.108} \\
\mathbb{E}[\ln p(w|\alpha)]_{w,\alpha} &=& -\frac{M}{2}\ln(2\pi) + \frac{M}{2}(\psi(a_N) - \ln b_N) \\
& & -\frac{a_N}{2b_N}[m_N^Tm_N + Tr(S_N)] \tag{10.109} \\
\mathbb{E}[\ln p(\alpha)]_\alpha &=& a_0\ln b_0 + (a_0 - 1)[\psi(a_N) - \ln b_N] \\
& & b_0\frac{a_N}{b_N} - \ln\Gamma(a_0) \tag{10.110} \\
-\mathbb{E}[\ln q(w)]_w &=& \frac{1}{2}\ln | S_N | + \frac{M}{2}[1 + \ln(2\pi)] \tag{10.111} \\
-\mathbb{E}[\ln q(\alpha)]_\alpha &=& \ln\Gamma(a_N) - (a_N - 1)\Psi(a_N) - \ln b_N + a_N \tag{10.112}
\end{eqnarray}
$$    

图10.9给出了下界$$ L(q) $$与多项式模型的阶数的关系图像，数据集是从一个三阶多项式中人工生成的。这里，先验参数被设置为$$ a_0 = b_0 = 0 $$，对应于无信息先验$$ p(\alpha) \propto 1 $$。根据2.3.6节的讨论，它是$$ \ln \alpha $$上的均匀分布。正如我们在10.1节看到的那样，$$ L $$表示模型的对数边缘似然函数$$ \ln p(t|M) $$的下界。因此，变分框架将最高的概率赋予了$$ M = 3 $$的模型。这与最大似然的结果相反。最大似然方法通过增加模型的复杂度尽可能地让误差变小，直到误差趋于0，这导致了最大似然方法倾向于选择具有严重过拟合现象的模型。
