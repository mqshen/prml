与误差函数的一阶导数情况一样，我们可以使用有限差来得到精度受数值计算的精度限制的二阶导数。如果我们对每对可能的权值施加一个扰动，那么我们得到

$$
\begin{eqnarray}
\frac{\partial^2 E}{\partial w_{ji}\partial w_{lk}} = \frac{1}{4\epsilon^2}\{E(w_{ji} + \epsilon, w_{lk} + \epsilon) - E(w_{ji} + \epsilon, w_{lk} - \epsilon) \\
-E(w_{ji} - \epsilon, w_{lk} + \epsilon) + E(w_{ji} - \epsilon, w_{lk} - \epsilon)\} + O(\epsilon^2) \tag{5.90}
\end{eqnarray}
$$

同样的，通过使用对称的中心差，我们确保残留误差是$$ O(\epsilon^2) $$而不是$$ O(\epsilon) $$。因为Hessian矩阵有$$ W^2 $$个元素，且每个元素的计算四次复杂度为$$ O(W) $$的前向传播（每个模式）。因此我们看到这种方法计算完整的Hessian矩阵需要$$ O(W^3) $$次操 作。所以，虽然在实际应用中它对于验证反向传播算法的正确性很有用，但是这个方法的计算性质很差。     


一个更加高效的数值导数的方法是将中心差应用于可以通过反向传播方法计算的一阶导数。得到

$$
\frac{\partial^2 E}{\partial w_{ji}\partial w_{lk}} = \frac{1}{2\epsilon}\left\{\frac{\partial E}{\partial w_{ji}}(w_{lk} + \epsilon) - \frac{\partial E}{\partial w_{ji}}(w_{ln} - \epsilon)\right\} + O(\epsilon^2) \tag{5.91}
$$

因为只有$$ W $$个权值需要加上扰动，且梯度可以通过$$ O(W) $$次计算得到，所以这种方法可以通过$$ O(W^2) $$次操作得到Hessian矩阵。    

