# SVM不同参数分类效果可视化分析

## 代码功能概述
这段代码展示了使用不同参数的RBF核SVM对二维数据进行分类，并通过子图网格可视化不同参数组合下的决策边界。主要功能包括：
- 加载并处理二维分类数据
- 定义多种SVM参数组合（C和gamma）
- 训练不同参数的SVM模型
- 可视化决策边界、间隔和支持向量
- 显示每个参数组合的分类准确率

## 代码详细解释

### 1. 导入必要的库
```python
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
```

### 2. 主程序入口
```python
if __name__ == "__main__":
```

### 3. 数据加载与准备
```python
data = pd.read_csv('data/svm3.txt', sep='\t', header=None)
x, y = data[[0, 1]], data[2]
```
- 从文本文件读取数据，制表符分隔
- 前两列作为特征(x)，第三列作为标签(y)

### 4. 参数组合定义
```python
clf_param = (('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),
             ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100),
             ('rbf', 1, 5), ('rbf', 50, 5), ('rbf', 100, 5), ('rbf', 1000,5))
```
- 每个元组包含三个元素：(核类型, C值, gamma值)
- 共12种参数组合：
  - 固定C=1，gamma变化：0.1, 1, 10, 100
  - 固定C=5，gamma变化：0.1, 1, 10, 100
  - 固定gamma=5，C变化：1, 50, 100, 1000

### 5. 创建可视化网格
```python
x1_min, x2_min = np.min(x, axis=0)
x1_max, x2_max = np.max(x, axis=0)
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)
```
- 计算数据范围
- 创建200×200的网格点
- 将网格点坐标存储在grid_test中

### 6. 设置绘图参数
```python
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])  # 背景色
cm_dark = mpl.colors.ListedColormap(['g', 'r'])  # 数据点颜色
mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 设置中文字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.figure(figsize=(14, 10), facecolor='w')  # 创建大图
```

### 7. 训练模型并可视化（核心部分）
```python
for i, param in enumerate(clf_param):
    # 创建SVM分类器
    clf = svm.SVC(C=param[1], kernel=param[0])
    clf.gamma = param[2]  # 设置gamma参数
    
    # 训练模型
    clf.fit(x, y)
    
    # 预测并计算准确率
    y_hat = clf.predict(x)
    title = u'C=%.1f，$\gamma$ =%.1f,准确率=%.2f' % (param[1], param[2], accuracy_score(y, y_hat))
    
    # 准备子图
    plt.subplot(3, 4, i+1)
    
    # 预测网格点分类
    grid_hat = clf.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    
    # 绘制决策区域
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
    
    # 绘制数据点
    plt.scatter(x[0], x[1], c=y, edgecolors='k', s=40, cmap=cm_dark)
    
    # 标记支持向量
    plt.scatter(x.loc[clf.support_, 0], x.loc[clf.support_, 1], 
                edgecolors='k', facecolors='none', s=100, marker='o')
    
    # 绘制决策边界和间隔
    z = clf.decision_function(grid_test)
    z = z.reshape(x1.shape)
    plt.contour(x1, x2, z, colors=list('kbrbk'), 
                linestyles=['--', '--', '-', '--', '--'],
                linewidths=[1, 0.5, 1.5, 0.5, 1], 
                levels=[-1, -0.5, 0, 0.5, 1])
    
    # 设置坐标范围和标题
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(title, fontsize=14)
```

#### 可视化元素说明
| 元素         | 说明                              | 代码实现                              |
| ------------ | --------------------------------- | ------------------------------------- |
| **背景色**   | 不同颜色表示不同类别的决策区域    | `plt.pcolormesh(...)`                 |
| **数据点**   | 原始数据点，颜色表示实际类别      | `plt.scatter(...)`                    |
| **支持向量** | 空心圆圈标记支持向量              | `plt.scatter(..., facecolors='none')` |
| **决策边界** | 实线表示决策边界（z=0）           | `plt.contour(levels=0)`               |
| **间隔边界** | 虚线表示间隔边界（z=-1和z=1）     | `plt.contour(levels=[-1,1])`          |
| **中间线**   | 虚线表示中间位置（z=-0.5和z=0.5） | `plt.contour(levels=[-0.5,0.5])`      |

### 8. 显示最终图表
```python
plt.suptitle(u'SVM不同参数的分类', fontsize=20)
plt.show()
```

## 参数对模型效果的影响

### 1. C值（正则化参数）
- **作用**：控制错误分类的惩罚程度
- **影响**：
  - 较小的C值：更大间隔，容忍更多错误分类（模型更简单）
  - 较大的C值：更小间隔，更少错误分类（模型更复杂）

### 2. gamma值（RBF核参数）
- **作用**：控制单个样本的影响范围
- **影响**：
  - 较小的gamma：决策边界更平滑，泛化能力更好
  - 较大的gamma：决策边界更复杂，可能过拟合

### 3. 参数组合效果分析

| 参数组合             | 效果分析                            |
| -------------------- | ----------------------------------- |
| **固定C，变化gamma** | 观察gamma对决策边界复杂度的影响     |
| **固定gamma，变化C** | 观察C值对间隔大小和错误容忍度的影响 |
| **C和gamma同时变化** | 观察参数间的交互作用                |

## 可视化效果说明

1. **决策区域**：背景色表示分类结果，浅绿色和浅红色分别代表两个类别
2. **数据点**：散点图显示实际数据分布，绿色和红色表示不同类别
3. **支持向量**：空心圆圈标记支持向量，这些点决定了决策边界
4. **决策边界**：
   - 实线：实际决策边界（z=0）
   - 虚线：间隔边界（z=-1和z=1）
   - 点虚线：中间位置（z=-0.5和z=0.5）
5. **标题信息**：显示当前参数组合和模型在训练集上的准确率

## 结果解读要点

1. **过拟合与欠拟合**：
   - 高gamma + 高C：复杂边界，可能过拟合
   - 低gamma + 低C：简单边界，可能欠拟合

2. **支持向量数量**：
   - 高C值：支持向量较少，决策边界更贴合数据
   - 低C值：支持向量较多，决策边界更平滑

3. **边界复杂度**：
   - 高gamma：决策边界更弯曲，适应数据细节
   - 低gamma：决策边界更平直，忽略细节

4. **准确率变化**：
   - 观察不同参数下准确率的变化
   - 注意训练集准确率与泛化能力的平衡

这种可视化方法能直观展示SVM参数如何影响模型性能，帮助选择最佳参数组合。