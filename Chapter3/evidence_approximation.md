在线性基函数模型的纯粹贝叶斯方法中，我们会引入超参数$$ \alpha, \beta $$的先验分布，然后通过对超参数以及参数$$ w $$求边缘化的方式来做预测。尽管我们可以解析的得到对$$ w $$的积分或对超参数的积分，但是求出对所有这些变量完整地边缘化的解析解是比较棘手的。这里我们介绍一种通过最大化对参数$$ w
$$积分得到的边缘似然函数，来确定超参数的具体值的近似方法。这个框架在统计学的文献中被称为经验贝叶斯（empirical Bayes）（Bernardo and Smith, 1994; Gelman et al., 2004），或第二类最大似然（type 2 maximum likelihood）（Berger, 1985），或广义最大似然（generalized maximum likelihood）。在机器学习文献中也被称为证据近似（evidence approximation）（Gull, 1989; MacKay, 1992a）。    

如果我们引入$$ \alpha, \beta $$上的超先验，那么预测分布可以通过边缘化$$ w,\alpha,\beta $$来获得：    

$$
p(t|\textbf{t})=\int\int\int p(t|w,\beta)p(w|\textbf{t},\alpha,\beta)p(\alpha,\beta|\textbf{t})dwd\alpha d\beta \tag{3.74}
$$

其中$$ p(t|w,\beta) $$由式（3.8）给出，$$ p(w|\textbf{t},\alpha,\beta) $$由式（3.49）给出，其中的$$ m_N, S_N $$分别由（3.53）（3.54）给出。为了让记号简洁，我们省略了对于输入变量$$ x $$的依赖。如果后验分布$$ p(\alpha,\beta|\textbf{t}) $$在$$ \hat{\alpha},\hat{\beta} $$周围是尖峰，那么把$$ \alpha,\beta $$固定为$$ \hat{\alpha},\hat{\beta} $$，然后对$$ w $$求积分：

$$
p(t|\textbf{t}) \simeq p(t|\textbf{t},\hat{\alpha},\hat{\beta}) = \int p(t|w,\hat{\beta})p(w|\textbf{t}, \hat{\alpha},\hat{\beta})dw \tag{3.75}
$$

就得到了预测分布。根据贝叶斯定理，$$ \alpha,\beta $$的后验分布由

$$
p(\alpha,\beta|\textbf{t}) \propto p(\textbf{t}|\alpha,\beta)p(\alpha,\beta) \tag{3.76}
$$

给出。如果先验相对较平，那么在证据框架中$$ \hat{\alpha},\hat{\beta} $$可以通过最大化边缘似然函数$$ p(\textbf{t}|\alpha,\beta) $$来获得。我们先得到线性基函数模型的边缘似然函数，然后再找出最大值。这将使我们能够不通过交叉验证，而直接从训练数据确定这些超参数的值。回忆一下比值$$ \alpha/\beta $$类似于正则化参数。    

此外，值得注意的是，如果在$$ \alpha,\beta $$上定义共轭（Gamma）先验分布，那么对式（3.74）中对这些超参数的边缘化可以通过$$ w $$上的学生t分布（见第2.3.7节）解析的计算出来。虽然得到的结果在$$ w $$上的积分不再有解析解，但是我们可以对这个积分求近似，例如，可以使用基于以后验概率分布的众数为中心的局部高斯近似的拉普拉斯近似方法(见第4.4节)，给证据框架提供了另一种实用的方法（Buntine and Weigend, 1991）。然而，作为关于$$ w
$$的被积函数的众数通常是强倾斜的，所以拉普拉斯近似方法不能描述概率质量中的大部分信息。这就导致最终的结果要比最大化证据的方法给出的结果差（MacKay, 1999）。    

回到证据框架中，我们注意到有两种方法可以用来最大化对数证据。我们可以解析的得到证据函数，然后令它的导数等于零，来得到了$$ \alpha,\beta $$的再估计方程（将在3.5.2节讨论）。另一种方法是，我们使用一种被称为期望最大化（EM）算法的技术，这将在9.3.4节讨论，并在那证明这两种方法会收敛到同一个解。    


