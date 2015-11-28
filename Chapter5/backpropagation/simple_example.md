上面的推导过程适用于一般的形式的误差函数，激活函数以及网络拓扑。为了阐述如何应用这个算法，我们来考虑一个特别的例子。我们选择了一个既简单又在实际应用中非常重要的例子，且在神经网络文献中许多应用都使用的这种类型的网络。具体地，我们会考虑图5.1展示的具有平方和误差，输出单元有线性激活函数即$$ y_k = a_k $$，隐藏单元具有由

$$
h(a) \equiv tanh(a)  \tag{5.58}
$$

给出的logistic sigmoid激活函数的二层网络。其中

$$
tanh(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}} \tag{5.59}
$$

这个函数的一个很有用的性质是它的导数可以表示为相当简单的形式。

$$
h'(a) = 1 - h(a)^2 \tag{5.60}
$$

我们也考虑标准平方和误差函数，那么模式$$ n $$的误差由

$$
E_n = \frac{1}{2}\sum\limits_{k=1}^K(y_k - t_k)^2 \tag{5.62}
$$

给出。其中对于一个特定的输入模式$$ x_n $$，$$ y_k $$是输出单元$$ k $$的激活，$$ t_k $$是对应的目标。    

对于训练集中的每个模式，首先我们使用

$$
\begin{eqnarray}
a_j = \sum\limits_{i=0}^Dw_{ji}^{(1)}x_i \tag{5.62} \\
z_j = tanh(a_j) \tag{5.63} \\
y_k = \sum\limits_{j=0}^Mw_{kj}^{(2)}z_j \tag{5.64} 
\end{eqnarray}
$$

来进行前向传播。然后使用

$$
\delta_k = y_k - t_k \tag{5.65}
$$

来计算每个输出的$$ \delta $$。接下来我们再利用获得的这些$$ \delta $$结合

$$
\delta_j = (1-z_j^2)\sum\limits_{k=1}^Kw_{kj}\delta_k \tag{5.66}
$$

来反向传播。
最终，第一层和第二层的关于权值的导数为：

$$
\frac{\partial E_n}{\partial w_{ji}^{(1)}} = \delta_jx_i , \frac{\partial E_n}{\partial w_{kj}^{(2)}} = \delta_kz_j \tag{5.67}
$$



