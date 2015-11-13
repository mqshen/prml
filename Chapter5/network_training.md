目前为止，我们把神经网络看成一大类从输入变量向量$$ x $$到输出变量向量$$ y $$的参数非线性函数。确定网络参数的一种简单方法类似于我们在1.1节对多项式曲线拟合问题的讨论：最小化平方和误差函数。给定一个由输入向量$$ \{x_n\} n = 1,...,N $$组成的训练集，以及对应的目标向量$$ t_n $$组成的集合，最小化误差函数    

$$
E(w) = \frac{1}{2}\sum\limits_{n=1}^N\Vert y(x_n,w) - t_n \Vert^2 \tag{5.11}
$$

然而，通过给网络的输出提供一个概率形式的表示，我们可以给网络训练提供一个更加一般的观点。在1.5.4节中我们已经看到使用概率预测的很多优势。这里它给我们输出单元的非线性函数及误差函数的选择提供一个清晰的动机。    

我们以讨论回归问题开始，现在我们只考虑可以取任何实数值的一元目标变量$$ t $$的情形。根据1.2.5与3.1节中的讨论，我们假设$$ t $$是均值依赖$$ x $$由神经网络的输出确定的高斯分布：    

$$
p(t|x, w) = \mathcal{N}(t|y(x,w),\beta^{-1}) \tag{5.12}
$$

其中$$ \beta $$是高斯噪声的精度（方差的逆）。当然，这种假设有些严格。在5.6节中，我们将会看到如何扩展这个方法来允许更一般的条件分布。对于（5.12）给出的条件分布，将输出单元激活函数取成恒等函数就足够了，因为这样的网络可以近似任何从$$ x $$到$$ y $$的连续函数。给定一个由$$ N $$个独立同分布的观测组成的数据集$$ X = {x_1,...,x_N} $$，以及对应的目标值$$ t =
{t_1,...,t_N} $$，我们可以构造对应的似然函数

$$
p(t|X,w,\beta) = \prod\limits_{n=1}^N p(t_n|x_n,w,\beta)
$$

取对数的负，得到误差函数

$$
\frac{\beta}{2}\sum\limits_{n=1}^N\{y(x_n,w)-t_n\}^2 - \frac{N}{2}\ln\beta + \frac{\beta}{2}\ln(2\pi) \tag{5.13}
$$

这可以用来学习参数$$ w, \beta $$。在5.7节，我们将会讨论神经网络的贝叶斯方法，现在我们考虑最大似然方法。注意，在神经网络的文献中，通常考虑最小化误差函数而不是最大化（对数）似然函数，因此这里我们遵循这个惯例。首先确定$$ w $$。最大化似然函数等价于最小化由

$$
E(w) = \frac{1}{2}\sum\limits_{n=1}^N\{y(x_n,w) - t_n\}^2 \tag{5.14}
$$

给出的平方和误差函数。其中我们已经去掉了加上和乘以的常数。由于最小化$$ E(w) $$得到的$$ w $$的值对应最大似然解，所以我们把它记作$$ w_{ML} $$。在实际应用中，网络函数$$ y(x_n, w) $$的非线性性导致误差$$ E(w) $$是非凸的，因此在实际应用中找到的可能是似然函数的局部最大值，对应于误差函数的局部最小值，这将在5.2.1节讨论。    

得到$$ w_{ML} $$后，$$ \beta $$的值可以通过最小化对数似然的负

$$
\frac{1}{\beta_{ML}} = \frac{1}{N}\sum\limits_{n=1}^N\{y(x_n,w_{ML}) - t_n}^2 \tag{5.15}
$$

注意，一旦我们寻找$$ w_{ML} $$的迭代最优化过程完成，就可以计算这个值。如果有多个目标变量，且假设给定$$ x, w $$的条件下，目标变量之间相互独立，且共享噪声精度$$ \beta $$，那么，目标变量的条件分布由

$$
p(t|x,w) = \mathcal{N}(t|y(x,w),\beta^{-1}I) \tag{5.16}
$$

给出。使用与一元目标变量的情形相同的推导过程，我们看到最大似然的权值由最小化平方和误差函数（5.11）确定。于是噪声的精度由

$$
\frac{1}{\beta_{ML}} = \frac{1}{NK}\sum\limits_{n=1}^N\Vert y(x_n,w_{ML}) - t_n \Vert^2 \tag{5.17}
$$

给出，其中$$ K $$是目标变量的数量。当没有独立性假设时，最优化问题变得稍微复杂一些。     

回忆一下，根据4.3.6节的讨论，我们看到在误差函数（负对数似然函数）和输出单元激活函数之间有一个自然的对应关系。在回归问题中，我们可以把神经网络看成具有一个恒等$$ y_k = a_k $$输出激活函数的模型。对应的平方和误差函数具有

$$
\frac{\partial E}{\partial a_k} = y_k - t_k \tag{5.18}
$$

性质。在5.3节讨论误差反向传播的时候，我们将会用到这几结果。    

现在考虑有单一目标变量$$ t $$，且$$ t = 1 $$表示类别$$ C_1 $$，$$ t = 0 $$表示类别$$ C_2 $$的二分类问题。根据4.3.6节中对标准链接函数的讨论，我们考虑一个具有单一输出，以logistic sigmoid函数作为激活函数的网络

$$
y = \sigma(a) \equiv \frac{1}{1+exp(-a)} \tag{5.19}
$$

所以$$ 0 \leq y(x,w) \leq 1 $$。我们可以把$$ y(x,w) $$解释为条件概率$$ p(C_1|x) $$，那么$$ p(C_2|x) $$的概率就是$$ 1 - y(x,w) $$。如果给定了输入。那么目标变量的条件概率分布是一个形式为

$$
p(t|x,w) = y(x,w)^t\{1-y(x,w)\}^{1-t} \tag{5.20}
$$

的伯努利分布。如果我们考虑一个由独立观测组成的训练集，那么由负对数给出的误差函数是形式为

$$
E(w) = -\sum\limits_{n=1}^N\{t_n\ln y_n + (1-t_n)\ln (1-y_n)\} \tag{5.21}
$$

的交叉熵误差函数。其中$$ y_n $$表示$$ y(x_n,w) $$。注意，由于我们假定目标值的标记都是正确的，所以这里没有与噪声精度$$ \beta $$相类似的东西。然而，模型可以容易的扩展到能够接受错误标记的情形。Simard et al.(2003)发现，对于分类问题，使用交叉熵误差函数的训练速度会比平方和误差函数更快，同时也提升了泛化能力。    

如果我们有$$ K $$个相互独立的二分类问题，那么我们可以使用具有$$ K $$个以logistic sigmoid函数作为激活函数的输出的神经网络。与每个输出相关联的是一个二元类别标签$$ t_k \in \{0,1\} k=1,...,K $$。如果我们假设类标签是独立的，那么给定输入向量，目标向量的条件概率分布为

$$
p(t|x,w) = \prod\limits_{k=1}^K y_k(x,w)^{t_k}[1-y_k(x,w)]^{1-t_k} \tag{5.22}
$$

取似然函数的负对数，可以得到误差函数

$$
E(w) = -\sum\limits_{n=1}^N\sum\limits_{k=1}^K\{t_{nk}\ln y_{nk} + (1 - t_{nk})\ln (1-y_nk)\} \tag{5.23}
$$

其中$$ y_{nk} $$表示$$ y_k(x_n, w) $$。与回归问题一样，对于指定输出单元，误差函数关于激活的导数的形式为式（5.18）。    

我们可以对比一下这个问题的神经网络解和第4章讨论的线性分类模型给出的解，来发现一些有趣的事情。假设我们使用图5.1所示的标准的两层神经网络。我们看到，网络第一层的权向量由各个输出所共享，而在线性模型中每个分类问题是独立地解决的。神经网络的第一层可以被看做进行了一个非线性的特征抽取，而不同的输出之间共享特征可以节省计算量，同时也提升了泛化能力。    

最后，我们考虑标准的多分类问题，其中每个输入被分到$$ K $$个互斥的类别中。二元目标变量$$ t_k \in \{0,1\} $$使用“1-of-K”编码方式，从而网络的输出可以表示为$$ y_k(x,w) = p(t_k = 1 | x) $$，因此误差函数为

$$
E(w) = -\sum\limits_{n=1}^N\sum\limits_{k=1}^K t_{nk}\ln y_k(x_n,w) \tag{5.24}
$$

根据4.3.4节的讨论，我们看到对应标准链接的输出单元激活函数有softmax函数：

$$
y_k(x,w) = \frac{exp(a_k(x,w))}{\sum\limits_jexp(a_j(x,w))} \tag{5.25}
$$

给出。其中满足$$ 0 \leq y_k \leq 1 $$和$$ \sum_ky_k = 1 $$。注意，给所有的$$ a_k(x,w) $$都加上一个常数，$$ y_k(x, w) $$是不变的，这就使得误差函数在权空间的某些方向上是常数。如果我们给误差函数加上一个恰当的正则化项（第5.5节），那么这种问题就可以避免。    

与之前一样，对于指定输出单元，误差函数关于激活的导数的函数形式为式（5.18）。    

总而言之，根据解决的问题的类型，输出单元激活函数和对应的误差函数，存在一个自然的选择。对于回归问题，我们使用线性输出，平方和误差函数，对于（多独立的）二分类问题，我们使用logistic sigmoid输出以及交叉熵误差函数，对于多类别分类问题，我们使用softmax输出以及对应的多分类交叉熵错误函数。对于涉及到两个类别的分类问题，我们可以使用单一的logistic sigmoid输出，也可以使用有两个输出的，且输出激活函数为softmax函数的神经网络。