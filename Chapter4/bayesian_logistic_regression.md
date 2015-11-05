现在我们以贝叶斯的观点来看logistic回归。logistic回归的精确贝叶斯推断是很难处理的。特别的，计算后验分布需要对先验分布与似然函数的乘积做标准化。而似然函数本身由每个数据点都有一个logistic sigmoid函数的乘积组成。预测分布的计算同样是很困难的。这里我们考虑使用拉普拉斯近似来处理贝叶斯logistic回归问题（Spiegelhalter and Lauritzen, 1990; MacKay, 1992b）。    


