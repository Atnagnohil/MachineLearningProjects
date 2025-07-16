# 🌲 梯度提升树（Gradient Boosting Tree）总结笔记

## ✅ 基本概念

**梯度提升树（Gradient Boosting Decision Tree, GBDT）** 是一种集成学习算法，通过不断叠加弱模型（通常是回归树）来优化一个损失函数。

它的基本思想是：

> 每一步都拟合损失函数对模型当前输出的**负梯度**（即模型的"**错误方向**"），然后加到模型上，相当于在函数空间中做梯度下降。

------

## 🧠 理解要点

### 📌 1. 提升树（Boosting Tree） vs 梯度提升树（GBDT）

- "提升树"：是 GBDT 的特例，损失函数是平方误差时，残差 = 负梯度
- GBDT：可以使用任意可导的损失函数（如对数损失、Huber损失等），残差不再是简单的 y−y^y - \hat{y}，而是对损失函数关于预测值的导数（梯度）

------

### 📌 2. 为什么使用“负梯度”？

- 在普通的梯度下降中，我们更新参数的方向是负梯度：
   θ←θ−η∂L∂θ\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
- 在 GBDT 中，我们不是优化参数，而是优化一个函数 F(x)F(x)，即预测模型本身：
   Fm(x)=Fm−1(x)−η⋅∇y^L(y,Fm−1(x))F_m(x) = F_{m-1}(x) - \eta \cdot \nabla_{\hat{y}} L(y, F_{m-1}(x))
- 每一步构建的树 hm(x)h_m(x) 拟合的就是负梯度：
   hm(x)≈−∂L∂F(x)h_m(x) \approx -\frac{\partial L}{\partial F(x)}

------

### 📌 3. 为什么使用“树”拟合负梯度？

- 现在你有一堆数据：xix_i 和对应的 −gi-g_i（负梯度）
- 这是一个典型的“回归问题” ～ 输入是 xx，输出是负梯度
- 回归树是处理这种非线性转换的最好工具
- 拟合后的树输出：告诉我们如何修正当前模型 Fm−1(x)F_{m-1}(x)

------

### 📌 4. 学习率（η\eta\uff09是干啥的？

- 控制每一步模型的“走步长度”
- 如果太大：可能导致过拟合或不稳定
- 如果太小：学习速度慢，需要进行更多次

> **如何确定 η\eta？**
>
> 通常作为超参数设置，常用值：0.01 ~ 0.3；
>  也可以在每一轮通过一维 line search 在验证集上找最优步长：
>
> η=arg⁡min⁡η∑iL(yi,Fm−1(xi)+ηhm(xi))\eta = \arg\min_\eta \sum_i L\left(y_i, F_{m-1}(x_i) + \eta h_m(x_i)\right)

------

## 🔁 梯度提升树的完整流程

1. 初始化模型：
    F0(x)=arg⁡min⁡c∑iL(yi,c)F_0(x) = \arg\min_c \sum_i L(y_i, c)
2. 对于每一轮 m=1,2,…,Mm = 1, 2, \dots, M：
   - 对每个样本 xix_i 计算负梯度：
      gi=−∂L(yi,F(xi))∂F(xi)∣F=Fm−1g_i = -\left.\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F = F_{m-1}}
   - 拟合一棵回归树 hm(x)h_m(x) 来近似 gig_i
   - 如果做 line search，选取步长：
      ηm=arg⁡min⁡η∑iL(yi,Fm−1(xi)+ηhm(xi))\eta_m = \arg\min_\eta \sum_i L\left(y_i, F_{m-1}(x_i) + \eta h_m(x_i)\right)
   - 更新模型：
      Fm(x)=Fm−1(x)+ηhm(x)F_m(x) = F_{m-1}(x) + \eta h_m(x)
3. 得到最终模型：
    FM(x)=∑m=1Mηhm(x)F_M(x) = \sum_{m=1}^{M} \eta h_m(x)

------

## ✅ 总结一句话

> GBDT 就是：**在函数空间里做梯度下降，每一步用一棵回归树去“修正预测”，沿着负梯度方向逻辑地近似目标函数。**

每棵树是一次“方向修正”，最终模型是它们的累加。