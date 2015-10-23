正如我们看到的，不管是高斯分布还是离散的输入，后验类概率密度都是由一般的线性模型和logistic sigmoid函数（$$K = 2$$个类别）或softmax（$$ K \geq 2 $$个类别）激活函数给出。这些都可以看成，当类条件密度$$ p(x|C_k) $$是指数族分布的成员时得到结果的特例。    

使用式（2.194）定义的指数族成员，得到$$ x $$分布的形式为

$$
p(x|\lambda_k) = h(x)g(\lambda_k)exp\{\lambda_k^Tu(x)\} \tag{4.83}
$$

现在，我们把注意力集中在$$ u(x) = x $$这样的分布上，然后使用式（2.236）引入一个缩放参数$$ s $$，那么我们就可以得到形式为

$$
p(x|\lambda_k,s) = \frac{1}{s}h\left(\frac{1}{s}x\right)g(\lambda_k)exp\left\{\frac{1}{s}\lambda_k^Tx\right\} \tag{4.84}
$$

注意，我们已经允许各个类别有自己的参数向量$$ \lambda_k $$，但是我们假设这些类别共享一个同样的伸缩参数$$ s $$。   

对于二分类问题，我们把这个类条件密度表达式代入式（4.58），得到类的后验概率同样由形式为

$$
a(x) = (\lambda_1 - \lambda_2)^Tx + \ln g(\lambda_1) - \ln g(\lambda_2) + \ln p(C_1) - \ln p(C_2) \tag{4.85}
$$

的线性函数$$ a(x) $$上的logistic sigmoid给出。    

同样的，对于$$ K $$个类别问题，我们把类条件密度代入式（4.63）得到

$$
a_k(x) = \lambda_k^Tx + \ln g(\lambda_k) + \ln p(C_k) \tag{4.86}
$$

这也是$$ x $$的线性函数。
