# _*_ coding: utf-8 _*_
'''
时间:      2025/7/20 16:35
@author:  andinm
'''

'''
绘制P-R 图
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay

# 设置随机种子确保结果可复现
np.random.seed(42)

# 1. 生成模拟数据集（400个样本，60%属于正类）
def generateData():
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=5,
        n_classes=2,
        weights=[0.4, 0.6],  # 40%负样本，60%正样本（模拟不平衡数据）
        random_state=42
    )
    return X, y


# 2. 分割数据集
def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def trainModel(X_train, X_test, y_train, y_test):
    # 3. 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 4. 获取预测概率（我们关注正类的概率）
    y_scores = model.predict_proba(X_test)[:, 1]

    # 5. 计算PR曲线数据
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    return precision, recall, thresholds, y_scores

'''
    代码中precision_recall_curve()函数实际上做了这样的事：
    
    thresholds = np.sort(np.unique(y_scores))[::-1]  # 获取所有唯一概率值并降序排列
    precision_list = []
    recall_list = []
    
    for threshold in thresholds:
        # 使用当前阈值进行二值化预测
        y_pred = (y_scores >= threshold).astype(int)
        
        # 计算当前阈值下的混淆矩阵
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        # 计算precision和recall
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 1
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_list.append(precision_val)
        recall_list.append(recall_val)
        
        对于预测概率中的每个唯一值（或经过优化的采样点）都作为一个阈值
        每个阈值会产生一对(recall, precision)值
        这些点连接起来就形成了PR曲线
    '''


def plot(precision, recall, thresholds, ix, baseline):
    # 7. 绘制PR曲线
    plt.figure(figsize=(10, 7))

    # 绘制PR曲线
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')


    plt.axhline(y=baseline, color='navy', linestyle='--',
                label=f'Baseline (Positive ratio = {baseline:.3f})')


    plt.scatter(recall[ix], precision[ix], marker='o', color='black',
                label=f'Optimal Threshold (F1 max at {thresholds[ix]:.2f})')

    # 设置图表属性
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (PPV)', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=15, pad=12)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)

    # 显示图表
    plt.tight_layout()
    plt.show()

# 8. 可选：使用sklearn内置绘图工具（另一种样式)
def plot2(pr_auc, y_test, y_scores):
    fig, ax = plt.subplots(figsize=(10, 7))
    PrecisionRecallDisplay.from_predictions(
        y_test, y_scores, ax=ax,
        name=f'Logistic Regression (AUC={pr_auc:.3f})'
    )
    ax.set_title('Precision-Recall Curve (sklearn display)', fontsize=15)
    ax.axhline(y=baseline, color='navy', linestyle='--', label='Baseline')
    ax.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

'''
关键指标:
- 正样本比例: 0.588
- PR曲线下面积 (AUC-PR): 0.846
- 最佳阈值: 0.274
- 最佳阈值下的精确率: 0.682
- 最佳阈值下的召回率: 0.957
- 最佳阈值下的F1分数: 0.796

'''

if __name__ == "__main__":
    X, y = generateData()
    X_train, X_test, y_train, y_test = splitData(X, y)

    precision, recall, thresholds, y_scores = trainModel(X_train, X_test, y_train, y_test)

    # 6. 计算曲线下面积 (AUC-PR)
    pr_auc = auc(recall, precision)
    # 添加随机分类器的基准线（正类比例）
    baseline = len(y_test[y_test == 1]) / len(y_test)
    # 标记最优阈值点（F1分数最大化的点）
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    '''通过F1-Score来计算模型最优的阈值
    （也可以通过BEP）'''
    ix = np.argmax(f1_scores)   # 取最大值


    # 9. 输出关键指标
    print("\n关键指标:")
    print(f"- 正样本比例: {baseline:.3f}")
    print(f"- PR曲线下面积 (AUC-PR): {pr_auc:.3f}")
    print(f"- 最佳阈值: {thresholds[ix]:.3f}")
    print(f"- 最佳阈值下的精确率: {precision[ix]:.3f}")
    print(f"- 最佳阈值下的召回率: {recall[ix]:.3f}")
    print(f"- 最佳阈值下的F1分数: {f1_scores[ix]:.3f}")
    plot(precision, recall, thresholds, ix, baseline)
    # plot2(pr_auc, y_test, y_scores)