##### 1.1
把（1.1）代入（1.2）中得到误差函数：$$ \sum\limits_{n=1}^{N} (w_0 + w_1x_n + ... + w_mx_n^m - t_n)^2 $$
设矩阵$$ B $$其中$$ B_{ij} = x_i^j $$我们的误差函数就变成的$$ (BW - y)^T(BW - y) $$，使其微分等于0得：$$ B^TBW = B^Ty $$整理可得

$$ \sum\limits_{j=0}^MA_{ij}w_j = T_i $$
其中
$$ A_{ij} = \sum\limits_{n=1}^N(x_n)^{i + j} , T_i = \sum\limits_{n=1}^N(x_n)^it_n $$

##### 1.2
代入（1.4）可得出差函数$$ (BW - y)^T(BW - y) + \frac{\lambda}{2}W^TW $$，使其微分等于0得：$$ (B^TB + \lambda I)W = B^Ty $$所以：
$$
\widetilde{A}_{ij} =A_{ij} + \lambda I_{ij}
$$

##### 1.3    
$$ p(a) = p(a|r)p(r) +  p(a|b)p(b) + p(a|g)p(g) = 0.34 $$
根据贝叶斯定理得：
$$ p(g|o) = \frac{p(o|g)p(g)}{p(o)} = \frac{0.3*0.6}{0.36} = 0.5 $$


