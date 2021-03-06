到目前为止，本书的关注的是密度估计和数据聚类这样的无监督学习。现在转而开始讨论回归这样的监督学习。回归的目的是预测$$ D $$维输入变量$$ x $$对应的一个或多个目标变量$$ t $$。在第一章中考虑多项式曲线拟合的时候已经碰到回归问题的例子。多项式是线性回归模型这一大类函数的一个特殊例子。它具有可调参数的线性函数的性质，并是这章的主要关注点。最简单的线性回归模型形式也是输入变量的线性函数。但是，通过将输入变量的非线性函数进行线性组合，可以得到一类被称为基函数（basis function）的更加有用的函数。这样的参数的线性函数模型，使其具有一些简单的分析性质，且关于输入变量是非线性的。    

给定一个包含$$ N $$个观测量$$ \{x_n\}, n = 1,...,N $$与对应的目标变量$$ \{t_n\} $$的训练数据集，目标是预测出新的$$ x $$值的目标变量$$ t $$。最简单的方法是，直接构造一个适当的函数$$ y(x) $$，能直接预测出新的输$$ x $$对应的$$ t $$的值。更一般地，从一个概率的观点来看，我们的目标是对表达了我们对每个$$ x $$值的目标$$ t $$的值的不确定性的预测分布$$ p(t|x)
$$进行建模。从这个条件分布中我们可以预测出每个新的$$ x $$对应的$$ t $$，这种方法等 同于最小化一个选择恰当的损失函数的期望值。就像在1.5.5节中讨论的那样，通常选择平方损失（squared loss）为实值变量的损失函数，这种情况下最优解由$$ t $$的条件期望给出。    

尽管，线性模型对于模式识别的实际应用来说有很大的局限性，特别是对于涉及到高维输入空间的问题，但它们有很好的分析性质，且组成了后续章节中讨论的更加复杂模型的基础。    

