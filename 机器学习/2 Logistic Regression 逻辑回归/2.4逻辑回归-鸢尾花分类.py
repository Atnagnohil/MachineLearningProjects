# _*_ coding: utf-8 _*_
'''
时间:      2025/6/2 22:55
@author:  andinm
'''
# 导入必要库
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import alpha
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 加载数据集
iris = load_iris()
# print(iris)
X_Orign = iris['data']
y = iris["target"]  # y设置为一位数组 防止警告
# print(X_Orign)
# print(y)

# 数据的含义
# print(iris.DESCR)
'''
---

.. _iris_dataset:

**鸢尾花植物数据集**
--------------------

**数据集特征:**

:样本数量: 150 (3个类别，每类50个样本)
:属性数量: 4个数值型预测属性和1个类别属性
:属性信息:
    - 花萼长度 (单位: 厘米)
    - 花萼宽度 (单位: 厘米)
    - 花瓣长度 (单位: 厘米)
    - 花瓣宽度 (单位: 厘米)
    - 类别:
            - 山鸢尾 (Iris-Setosa)
            - 杂色鸢尾 (Iris-Versicolour)
            - 维吉尼亚鸢尾 (Iris-Virginica)

:汇总统计信息:

============== ===== ===== ======= ===== ====================
                最小值  最大值   平均值   标准差   类别相关性
============== ===== ===== ======= ===== ====================
花萼长度:        4.3   7.9    5.84    0.83     0.7826
花萼宽度:        2.0   4.4    3.05    0.43    -0.4194
花瓣长度:        1.0   6.9    3.76    1.76     0.9490  (高!)
花瓣宽度:        0.1   2.5    1.20    0.76     0.9565  (高!)
============== ===== ===== ======= ===== ====================

:缺失属性值: 无
:类别分布: 3个类别各占33.3%。
:创建者: R.A. 费舍尔 (R.A. Fisher)
:提供者: 迈克尔·马歇尔 (Michael Marshall, MARSHALL%PLU@io.arc.nasa.gov)
:日期: 1988年7月

著名的鸢尾花数据库，最初由 R.A. 费舍尔爵士使用。该数据集取自费舍尔的论文。请注意，它与 R 语言中的数据集相同，但与 UCI 机器学习知识库中的版本不同，后者包含两个错误的数据点。

这可能是模式识别文献中最广为人知的数据库。费舍尔的论文是该领域的经典之作，至今仍被频繁引用（例如参见 Duda & Hart 的著作）。该数据集包含 3 个类别，每个类别有 50 个样本，
每个类别代表一种鸢尾花植物。其中一个类别与其他两个类别是线性可分的；而后两个类别彼此之间则**不是**线性可分的。

.. dropdown:: 参考文献

  - Fisher, R.A. "The use of multiple measurements in taxonomic problems"（多变量测量在分类学问题中的应用）《优生学年刊》(Annual Eugenics), 7, Part II, 179-188 (1936); 亦收录于《数理统计贡献》(Contributions to Mathematical Statistics) (John Wiley, NY, 1950).
  - Duda, R.O., & Hart, P.E. (1973) 《模式分类与场景分析》(Pattern Classification and Scene Analysis). (Q327.D83) John Wiley & Sons. ISBN 0-471-22361-1. 见第 218 页.
  - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System Structure and Classification Rule for Recognition in Partially Exposed Environments"（在局部暴露环境中识别的新系统结构和分类规则）《IEEE 模式分析与机器智能汇刊》(IEEE Transactions on Pattern Analysis and Machine Intelligence), Vol. PAMI-2, No. 1, 67-71.
  - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule"（缩减最近邻规则）《IEEE 信息论汇刊》(IEEE Transactions on Information Theory), May 1972, 431-433.
  - 亦可参见: 1988 MLC 会议论文集, 54-64. Cheeseman 等人的 AUTOCLASS II 概念聚类系统在数据中发现了 3 个类别。
  - 更多相关文献...'''

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X_Orign)
# print(X)

# 选取两个特征可视化
feature1 = 2
feature2 = 3

plt.scatter(X[0:50, feature1], X[0:50, feature2], color="red", marker='o', label='setosa')              # 前50个数据                - 山鸢尾 (Iris-Setosa)
plt.scatter(X[50:100, feature1], X[50:100, feature2], color="blue", marker='x', label='Versicolour')    # 50到100个数据             - 杂色鸢尾 (Iris-Versicolour)
plt.scatter(X[100:150, feature1], X[100:150, feature2], color="green", marker='+', label='Virginica')   # 100到150个数据            - 维吉尼亚鸢尾 (Iris-Virginica)
plt.legend(loc="upper left")
plt.grid(True, alpha=0.7)
plt.show()


# 将整个数据训练模型
model1 = linear_model.LogisticRegression(C=50, max_iter=2000)
model1.fit(X, y)
print(model1.intercept_)
print(model1.coef_)
y_hat1 = model1.predict(X)
print(f"在训练集上的精准度{accuracy_score(y_hat1, y):.2%}")

# 选取两个特征训练模型，以便可视化
feature1 = 2
feature2 = 3
model2 = linear_model.LogisticRegression(C=50, max_iter=2000)
model2.fit(X[:, feature1:feature2+1], y)
print(model2.intercept_)
print(model2.coef_)
y_hat2 = model2.predict(X[:, feature1:feature2+1])
print(f"在训练集上的精准度{accuracy_score(y_hat2, y):.2%}")

# 可视化结果
def plotDescisionBoundary(X, feature1, feature2, model2):
    # meshgrid函数生成两个网格矩阵
    h = 0.02
    x_min, x_max = X[:, feature1].min() - 0.5, X[:, feature1].max() + 0.5
    y_min, y_max = X[:, feature2].min() - 0.5, X[:, feature2].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # print(xx, yy)
    # print(np.c_[xx.ravel(), yy.ravel()])
    z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, z, cmap=plt.cm.Paired)

    plt.scatter(X[0:50, feature1], X[0:50, feature2], color='red', marker='o', label='setosa')  # 前50个样本
    plt.scatter(X[50:100, feature1], X[50:100, feature2], color='blue', marker='x', label='versicolor')  # 中间50个
    plt.scatter(X[100:, feature1], X[100:, feature2], color='green', marker='+', label='Virginica')  # 后50个样本
    plt.show()
plotDescisionBoundary(X, feature1, feature2, model2)













