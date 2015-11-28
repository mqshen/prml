我们已经知道，让模型对于一组变换具有不变性的一种方法是使用原始输入模式的变换后的模式来扩展训练集。这里，我们说明，这种方法与切线传播的方法密切相关（Bishop, 1995b; Leen, 1995）。    

与5.5.4节一样，我们要考虑由单一参数$$ \xi $$控制的变换，且这个变换由函数$$ s(x,\xi) $$描述，其中$$ s(x, 0) = x $$。我们也会考虑平方和误差函数。对于未经过变换的输入，误差函数可以写成（在无限数据集的极限情况下）

$$
E = \frac{1}{2}\int\int\{y(x) - t\}^2p(t|x)p(x)dxdt \tag{5.129}
$$

正如1.5.5节讨论的那样。这里，为了保持记号的简洁，我们考虑有一个输出单元的网络。如果我们现在考虑每个数据点的无穷多个由参数为$$ \xi $$的变换施加了扰动的副本，其中$$ \xi $$服从概率分布$$ p(\xi) $$，那么在这个扩展的误差函数上定义的误差函数可以写成

$$
\tilde{E}=\frac{1}{2}\int\int\int\{y(s(x,\xi)) - t\}^2p(t|x)p(x)p(\xi)dxdtd\xi \tag{5.130}
$$

现在，我们假设分布$$ p(xi) $$的均值为0，且方差很小，即我们只考虑对原始输入向量的小的变换。对变换函数关于$$ \xi $$做泰勒展开得到

$$
\begin{eqnarray}
x(x,\xi) &=& s(x,0) + \xi\left.\frac{\partial}{\partial \xi}s(x,\xi)\vphantom{\Big|}\right| _{\xi = 0} + \frac{\xi^2}{2}\left.\frac{\partial^2}{\partial \xi^2}s(x,\xi)\vphantom{\Big|}\right| _{\xi = 0} + O(\xi^3) \\
&=& x + \xi\tau + \frac{1}{2}\xi^2\tau' + O(\xi^3)
\end{eqnarray}
$$

其中$$ \tau' $$表示$$ s(x,\xi) $$关于$$ \xi $$的二阶导数在$$ \xi = 0 $$处的值。这使得我们可以展开模型函数得到    

$$
y(s(x,\xi)) = y(x) + \xi\tau^T\nabla y(x) + \frac{\xi^2}{2}\left[(\tau')^T\nabla y(x) + \tau^T\nabla\nabla y(x)\tau\right] + O(\xi^3)
$$

代入平均误差函数（5.130）得到

$$
\begin{eqnarray}
\tilde{E} &=& \frac{1}{2}\int\int\{y(x) - t\}^2p(t|x)p(x)dxdt \\
&+& \mathbb{E}[\xi]\int\int\{y(x) - t\}\tau^T\nabla y(x)p(t|x)p(x)dxdt \\
&+& \mathbb{E}[\xi^2]\frac{1}{2}\int\int\Bigg[\{y(x) - t\}\left\{(\tau')^T\nabla y(x) + \tau^T\nabla\nabla y(x)\tau\right\} \\
&+& \left(\tau^T\nabla y(x)\right)^2\Bigg]p(t|x)p(x)dxdt + O(\xi^3)
\end{eqnarray}
$$

由于变换的分布的均值为0，因此我们有$$ \mathbb{E}[\xi] = 0 $$。且，我们把$$ \mathbb{E}[\xi^2] $$记作$$ \lambda $$。省略$$ O(\xi^3) $$项，这样平均误差函数就变成了    

$$
\tilde{E} = E + \lambda\omega \tag{5.131}
$$

其中$$ E $$是原始的平方和误差，正则化项$$ \omega $$的形式为

$$
\begin{eqnarray}
\omega &=& \frac{1}{2}\int\Bigg[\{y(x) - \mathbb{E}[t|x]\}\left\{(\tau')^T\nabla y(x) + \tau^T\nabla\nabla y(x)\tau\right\} \\
&+& \left(\tau^T\nabla y(x)\right)^2\Bigg]p(x)dx \tag{5.132}
\end{eqnarray}
$$

其中我们已经对$$ t $$进行了积分。    

我们可以进一步简化这个正则化项，如下所述。在1.5.5节，我们已经看到，使平方和误差函数达到最小值的函数为目标值$$ t $$的条件均值$$ \mathbb{E}[t|x] $$。根据式（5.131），我们看到正则化的误差函数等于非正则化的误差函数加上一个$$ O(\xi^2) $$的项，因此最小化总误差函数的网络函数的形式为    

$$
y(x) = \mathbb{E}[t|x] + O(\xi^2) \tag{5.133}
$$

从而，正则化项中的第一项消失，剩下的项为

$$
\omega = \frac{1}{2}\int\left(\tau^T\nabla y(x)\right)^2p(x)dx \tag{5.134}
$$

这等价于切线传播的正则化项（5.128）。    

如果我们考虑一个特殊情况，即输入变量的变换只是简单地添加随机噪声$$ x \to x + xi $$，那么正则化项的形式为    

$$
\omega = \frac{1}{2}\int\Vert \nabla y(x) \Vert^2p(x)dx \tag{5.135}
$$

这被称为Tikhonov正则化（Tikohonov and Arsenin, 1977; Bishop, 1995b）。这个正则化项关于网络权值的导数可以使用扩展的反向传播算法求出（Bishop, 1993）。我们看到，对于小的噪声，Tikhonov正则化与对输入添加随机噪声有关系。可以证明，在恰当的情况下，这种做法会提升模型的泛化能力。
