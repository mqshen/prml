在第1章中，通过最小化平方和误差函数使多项式函数来拟合数据集。同样展示了这个误差函数是高斯噪声模型下最大似然解的自然结果。让我们回到这个讨论中，并考虑最小二乘方法，并更加详细的讨论它与最大似然方法的关系。    

和之前一样，假设目标变量$$ t $$是由确定函数$$ y(x, w) $$加上高斯噪声给出的：    

$$
t = y(x,w) + \epsilon \tag{3.7}
$$

其中$$ \epsilon $$是均值为0，精度（方差的逆）为$$ \beta $$的高斯随机变量。因此可以写成：    

$$
p(t|x,w,\beta) = \mathcal{N}(t|y(x,w), \beta^{-1}) \tag{3.8}
$$

回忆一下，如果假设一个平方损失函数，那么新的$$ x $$的值的最优的预测是由目标变量的条件均值给出。在式（3.8）的高斯条件分布下，得到条件均值：    

$$
\mathbb{E}[t|x] = \int tp(t|x)dt = y(x,w) \tag{3.9}
$$

注意，高斯噪声隐含$$ x $$上的$$ t $$的条件分布是单峰的，这可能不适用于某些应用。一个混合的条件高斯分布扩展，允许多峰条件分布，这我们将在14.5.1节中讨论。    

现在，考虑输入$$ X = \{x_1,...,x_N\} $$和对应的目标值$$ t_1,...,t_N $$的数据集。把由目标向量$$ \{t_n\} $$组成列向量，记作$$ \textbf{t} $$。其中选择这个字体是为了与多元目标值的一次观测，记作$$ t $$做区分。假设这些数据是从分布（3.8）中独立的取出。那么，得到可调节参数$$ w, \beta $$的最大似然函数，形式为：    

$$
p(\textbf{t}|\textbf{X,w},\beta) = \prod\limits_{n=1}^N\mathcal{N}(t_n|w^T\phi(x_n),\beta^{-1}) \tag{3.10}
$$

其中使用了式（3.3）。注意，在监督学习问题如回归（或分类）中，我们不是为了寻找输入变量分布的模型。所以$$ x $$会一直出现在条件变量的位置，因此从现在开始，为了保持记号的简洁性，在诸如$$ p(\textbf{t}|x,w,\beta) $$这样的表达式中不显式地写出$$ x $$。取似然函数的对数，并使用一元高斯的标准形式（2.146），得到：    

$$
\begin{eqnarray}
\ln p(\textbf{t}|w, \beta) &=& \sum\limits_{n=1}^N \ln\mathcal{N}(t_n|w^T\phi(x_n),\beta^{-1}) \\
&=& \frac{N}{2}\ln\beta-\frac{N}{2}\ln(2\pi) - \beta E_D(w) \tag{3.11}
\end{eqnarray}
$$

其中平方和误差函数定义为：     

$$
E_D(w) = \frac{1}{2}\sum\limits_{n=1}^N\{t_n - w^T\phi(w_n)\}^2 \tag{3.12}
$$

已经得到似然函数，我们可以通过最大似然的方法来确定$$ w, \beta $$。首先对于$$ w $$最大化。正如我们已经在1.2.5节中已经看到的那样，我们看到在条件高斯噪声分布的情况下，线性模型的最大化似然 函数等价于最小化由$$ E_D(w) $$给出平方和误差函数。式（3.11）给出的对数似然函数的梯度为：    

$$
\nabla\ln p(\textbf{t}|w,\beta) = \sum\limits_{n=1}^N\{t_n - w^T\phi(x_n)\}\phi(x_n)^T \tag{3.13}
$$

使这个梯度等于0，得到：    

$$
0 = \sum\limits_{n=1}^Nt_n\phi(x_n)^T - w^T\left(\sum\limits_{n=1}^N\phi(x_n)\phi(x_n)^T\right) \tag{3.14}
$$

求解w，得到：    

$$
w_{ML} = (\Phi^T\Phi)^{-1}\Phi^T\textbf{t} \tag{3.15}
$$

这被称为最小二乘问题的正规方程组（normal equations）。其中$$ \Phi $$是被称为设计矩阵（design matrix）的一个$$ N \times M $$的矩阵，其中$$ \Phi_{nj} = \phi_j(x_n) $$，即

$$
\begin{eqnarray}
\Phi = 
\left(
\begin{array}{cccc}
\phi_0(x_1) & \phi_1(x_1) & \cdots & \phi_{M-1}(x_1) \\
\phi_0(x_2) & \phi_1(x_2) & \cdots & \phi_{M-1}(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_0(x_N) & \phi_1(x_N) & \cdots & \phi_{M-1}(x_N) 
\end{array}
\right) \tag{3.16}
\end{eqnarray}
$$

量

$$
\Phi^+ \equiv (\Phi^T\Phi)^{-1}\Phi^T \tag{3.17}
$$

被称为矩阵$$ \Phi $$的摩尔彭罗斯伪逆（Moore-Penrose pseudo-inverse）（Rao and Mitra, 1971; Golub and Van Loan, 1996）。它是逆矩阵概念在非方阵的推广。实际上，如果$$ \Phi $$是方阵且可逆，那么使用性质$$ (AB)^{−1} = B^{−1}A^{−1} $$，可以得到$$ \Phi^+ \equiv \Phi^{−1} $$。    

现在，我们可以更加深刻地认识偏置参数$$ w_0 $$。如果显式地写出偏置参数，那么误差函数（3.12）变为：    

$$
E_D(w) = \frac{1}{2}\sum\limits_{n=1}^N\{t_n - w_0 - \sum\limits_{j=1}^{M-1}w_j\phi_j(x_n)\}^2 \tag{3.18}
$$

对于$$ w_0 $$求导并使其等于0，求解$$ w_0 $$可得：    

$$
w_0 = \bar{t} - \sum\limits_{j=1}^{M-1}w_j\bar{\phi}_j \tag{3.19}
$$

其中定义了：    

$$
\bar{t} = \frac{1}{N}\sum\limits_{n=1}^Nt_n , \bar{\phi}_j = \frac{1}{N}\sum\limits_{n=1}^N\phi_j(x_n) \tag{3.20}
$$

因此偏置$$ w_0 $$补偿了目标值的均值(在训练集上的)与基函数的值的加权均值之间的差。    

我们也可以对于噪声精度参数$$ \beta $$最大化对数似然函数（3.11），得到：    

$$
\frac{1}{\beta_{ML}} = \frac{1}{N}\sum\limits_{n=1}^N\{t_n - w_{ML}^T\phi(x_n)\}^2 \tag{3.21}
$$

因此，我们看到噪声精度的逆是由目标值在回归函数周围的残差的方差给出。



