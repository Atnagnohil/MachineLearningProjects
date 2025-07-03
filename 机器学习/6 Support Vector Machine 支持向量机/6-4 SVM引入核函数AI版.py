# _*_ coding: utf-8 _*_
'''
时间:      2025/7/3 20:21
@author:  andinm
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
import random


plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

def loadData(fileName):
    """
    从指定文件加载数据。
    文件中每一行应包含两个特征和一个标签，以制表符分隔。
    """
    dataMatIn = []; classLabels = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMatIn.append([float(lineArr[0]), float(lineArr[1])])
        classLabels.append(float(lineArr[2]))
    fr.close()
    return dataMatIn,classLabels

def chooseJrand(i, m):
    """
    从0到m-1之间随机选择一个不等于i的整数j。
    用于SMO算法中选择第二个alpha。
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(alphaJNewUnClip, L ,H):
    """
    将alpha值裁剪到[L, H]范围内。
    """
    if alphaJNewUnClip > H:
        return H
    elif L <= alphaJNewUnClip <= H:
        return alphaJNewUnClip
    else:
        return L

def kernelTrans(X, A, kTup):
    """
    计算核变换（核函数）。
    参数:
        X (矩阵): 数据矩阵 (m x n)。
        A (矩阵): 单个数据点 (1 x n) 或另一个数据矩阵。
        kTup (元组): 指定核类型和参数的元组，例如 ('lin', 0) 表示线性核，('rbf', k1) 表示RBF核。
    返回:
        K (矩阵): 核矩阵或向量。
    """
    m, n = np.shape(X)
    K = np.asmatrix(np.zeros((m, 1)))
    if kTup[0] == 'lin':  # 线性核
        K = X @ A.T
    elif kTup[0] == 'rbf':  # 高斯核
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow @ deltaRow.T
        K = np.exp(K / (-2 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized') # 未识别的核函数
    return K

class OptStruct(object):
    """
    一个数据结构，用于保存SMO算法所需的所有变量，
    包括预计算的核矩阵。
    """
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        # 确保输入数据是NumPy矩阵
        self.dataMat = np.asmatrix(dataMatIn) # 数据矩阵（特征）
        self.labelMat = np.asmatrix(classLabels).reshape(-1, 1) # 标签
        self.C = C # 惩罚参数
        self.tol = toler # KKT条件违反的容忍度
        self.m = np.shape(self.dataMat)[0] # 数据点数量
        self.alphas = np.asmatrix(np.zeros((self.m, 1))) # Alpha乘子
        self.b = 0 # 偏置项
        # eCache 存储每个数据点的误差 (Ei)。
        # 第一列是有效标志（1表示有效，0表示无效），
        # 第二列存储实际的误差值。
        self.eCache = np.asmatrix(np.zeros((self.m, 2)))
        self.K = np.asmatrix(np.zeros((self.m, self.m))) # 预计算的核矩阵
        self.kTup = kTup # 核类型和参数

        # 预计算所有训练数据点对之间的核矩阵
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.dataMat, self.dataMat[i, :], kTup)

def calcEk(oS, k):
    """
    使用预计算的核矩阵计算给定数据点k的误差。
    Ek = f(xk) - yk
    其中 f(xk) = sum(alpha_i * y_i * K(x_i, x_k)) + b
    """
    # 使用预计算的核矩阵oS.K计算f(xk)
    # oS.K[:, k] 给出列向量 [K(x_0, x_k), K(x_1, x_k), ..., K(x_m-1, x_k)]
    fXk_term = (np.multiply(oS.alphas, oS.labelMat).T @ oS.K[:, k]).item()
    fXk = fXk_term + oS.b

    # 获取真实标签yk，确保它是标量。
    label_k_scalar = oS.labelMat[k].item()

    # 计算误差Ek
    Ek = fXk - label_k_scalar
    # 显式返回一个标准的Python浮点数，以防止DeprecationWarning
    return float(Ek)

def chooseJ(i, oS, Ei):
    """
    选择第二个alpha (alpha_j) 进行优化。
    优先选择使 |Ei - Ej| 最大的j。
    如果缓存中没有有效的Ej，则随机选择一个j。
    """
    maxj, Ej, maxDeltaE = -1, 0, 0
    # 将Ei标记为有效并将其值存储在缓存中。
    oS.eCache[i, 0] = 1
    # 使用 .item() 确保存储的是标量值，处理潜在的NumPy数组输入
    oS.eCache[i, 1] = Ei.item() if hasattr(Ei, 'item') else Ei

    # 获取有效（非零标志）误差缓存条目的索引。
    validEcacheList = np.nonzero(oS.eCache[:, 0])[0]

    # 如果缓存中有有效条目（不仅仅是当前的i）
    if (len(validEcacheList)) > 1:
        for j in validEcacheList:
            if j == i: continue # 如果j与i相同则跳过
            Ek = calcEk(oS, j) # 计算Ej（即索引j的Ek）
            deltaE = abs(Ei - Ek) # 计算差值
            # 如果此差值大于当前最大值，则更新。
            if deltaE > maxDeltaE:
                maxj = j
                maxDeltaE = deltaE
                Ej = Ek
        return maxj, Ej
    else: # 如果缓存中没有有效条目，则随机选择j。
        j = chooseJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    """
    更新数据点k的误差缓存。
    """
    Ek = calcEk(oS, k)
    oS.eCache[k, 0] = 1 # 标记为有效
    # 使用 .item() 确保存储的是标量值，处理潜在的NumPy数组输入
    oS.eCache[k, 1] = Ek.item() if hasattr(Ek, 'item') else Ek

def innerL(oS, i):
    """
    SMO算法的内循环。尝试优化alpha_i和alpha_j。
    如果成功更新了一对alpha，则返回1，否则返回0。
    """
    Ei = calcEk(oS, i)

    # 检查KKT条件：
    # 如果当前的alpha_i违反了KKT条件，则尝试优化它。
    # 条件1: (label_i * Ei < -toler) 且 (alpha_i < C) -> alpha_i 可以增加
    # 条件2: (label_i * Ei > toler) 且 (alpha_i > 0) -> alpha_i 可以减少
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
       ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):

        j,Ej = chooseJ(i, oS, Ei) # 选择第二个alpha (alpha_j)
        alphaIOld, alphaJOld = oS.alphas[i].copy().item(), oS.alphas[j].copy().item()

        # 使用预计算的核矩阵oS.K计算核项
        K11 = oS.K[i, i]
        K22 = oS.K[j, j]
        K12 = oS.K[i, j]
        K21 = oS.K[j, i]

        eta = K11 + K22 - 2 * K12 # alpha_j更新的分母

        # 如果eta <= 0，则优化步骤无效（Hessian矩阵非正定）
        if eta <= 0:
            return 0

        # 计算未裁剪的新alpha_j
        alphaJNewUncilp = alphaJOld + (oS.labelMat[j] * (Ei - Ej) / eta)

        # 确定alpha_j的上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, alphaJOld - alphaIOld)
            H = min(oS.C, oS.C + alphaJOld - alphaIOld)
        else:
            L = max(0, alphaJOld + alphaIOld - oS.C)
            H = min(oS.C, alphaJOld + alphaIOld)

        if L==H:
            return 0

        # 将alpha_j裁剪到其边界[L, H]内
        alphaJNew = clipAlpha(alphaJNewUncilp, L, H)

        # 根据新的alpha_j计算新的alpha_i
        alphaINew = alphaIOld + oS.labelMat[i] * oS.labelMat[j] * (alphaJOld - alphaJNew)

        # 更新OptStruct中的alphas
        oS.alphas[i] = alphaINew
        oS.alphas[j] = alphaJNew

        # 更新i和j的误差缓存
        updateEk(oS, i)
        updateEk(oS, j)

        # 检查alpha_j是否显著移动。如果没有，则没有显著更新。
        if (abs(oS.alphas[j] - alphaJOld) < 0.00001):
            return 0

        # 更新偏置项b
        b1New = (-Ei - oS.labelMat[i] * K11 * (alphaINew - alphaIOld) -
                 oS.labelMat[j] * K21 * (alphaJNew - alphaJOld) + oS.b)
        b2New = (-Ej - oS.labelMat[i] * K12 * (alphaINew - alphaIOld) -
                 oS.labelMat[j] * K22 * (alphaJNew - alphaJOld) + oS.b)

        # 根据alpha_i或alpha_j是否在边界上设置b
        if alphaINew > 0 and alphaINew < oS.C:
            oS.b = b1New
        elif alphaJNew > 0 and alphaJNew < oS.C:
            oS.b = b2New
        else: # 两者都在边界上或都为0，取平均值
            oS.b = (b1New + b2New) / 2.0
        return 1  # 表示更新了一对alpha
    else:
        return 0 # 未发生更新

def smoPro(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
    用于SVM训练的简化SMO算法。
    迭代优化alpha乘子和偏置项。
    支持线性核和RBF核。
    """
    # 将dataMatIn和classLabels转换为NumPy矩阵以用于OptStruct
    oS = OptStruct(np.asmatrix(dataMatIn), np.asmatrix(classLabels).reshape(-1, 1), C, toler, kTup)
    iter = 0
    entireSet = True # 标志，指示是遍历整个数据集还是只遍历非边界alpha
    alphaPairsChanged = 0 # 一次迭代中alpha对更新的计数器

    # 主优化循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # 遍历全部数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(oS, i)
                print("完整数据集, 迭代次数: %d, i: %d, alpha对改变数量: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历非边界数据集
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(oS, i)
                print("非边界数据集, 迭代次数: %d, i: %d, alpha对改变数量: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 在遍历整个集合和非边界集合之间切换
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0): # 如果非边界集合中没有alpha改变，则下一次迭代遍历完整集合
            entireSet = True
        print("当前迭代次数: %d" % iter)
    return oS.b, oS.alphas, oS # 返回oS对象用于绘图

def predict_decision_function(oS, x_new):
    """
    使用训练好的SVM计算新数据点x_new的决策函数值。
    f(x_new) = sum(alpha_i * y_i * K(x_i, x_new)) + b
    """
    # 确保x_new是用于kernelTrans的矩阵
    x_new_mat = np.asmatrix(x_new).reshape(1, -1) # 确保它是一个1xn矩阵
    # 计算所有训练数据点与x_new之间的核值
    K_new = kernelTrans(oS.dataMat, x_new_mat, oS.kTup) # K_new将是m x 1

    # 求和 alpha_i * y_i * K(x_i, x_new)
    fX = (np.multiply(oS.alphas, oS.labelMat).T @ K_new).item() + oS.b
    return fX

def plotPoint(dataMatIn, classLabels):
    """
    绘制原始数据点，按其类别标签着色。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['g', 'r']) # 绿色代表一类，红色代表另一类
    # 散点图：x坐标来自第一列，y坐标来自第二列
    # 颜色由classLabels决定
    ax.scatter(np.array(dataMatIn)[:, 0], np.array(dataMatIn)[:, 1],
               c=np.array(classLabels).squeeze(), cmap=cm_dark, s=30)
    plt.title("原始数据点 (Original Data Points)")
    plt.show()

def plotDesignSurface(dataMat, classLabels, oS):
    """
    绘制数据点和学习到的决策边界，适用于线性核和非线性核。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(np.array(dataMat)[:, 0], np.array(dataMat)[:, 1],
               c=np.array(classLabels).squeeze(), cmap=cm_dark, s=30)

    # 创建用于绘制决策边界的网格
    # 根据数据调整最小/最大值以确保良好覆盖
    x_min, x_max = np.array(dataMat)[:, 0].min() - 0.5, np.array(dataMat)[:, 0].max() + 0.5
    y_min, y_max = np.array(dataMat)[:, 1].min() - 0.5, np.array(dataMat)[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), # 更精细的步长以获得更平滑的曲线
                         np.arange(y_min, y_max, 0.05))

    # 预测网格中每个点的决策函数
    # np.c_ 组合列，然后重塑为 (N, 2)
    Z = np.array([predict_decision_function(oS, np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界（Z=0的等高线）
    ax.contour(xx, yy, Z, levels=[0], colors='blue', linewidths=2)
    # 绘制区域（contourf）- levels定义了着色的区域
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['lightgreen', 'lightblue'], alpha=0.4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.title("决策平面 (Decision Surface)")
    plt.show()

def plotSupportVector(dataMat, classLabels, oS):
    """
    绘制数据点、决策边界并突出显示支持向量。
    支持向量是alpha值非零的数据点。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(np.array(dataMat)[:, 0], np.array(dataMat)[:, 1],
               c=np.array(classLabels).squeeze(), cmap=cm_dark, s=30)

    # 创建用于绘制决策边界的网格
    x_min, x_max = np.array(dataMat)[:, 0].min() - 0.5, np.array(dataMat)[:, 0].max() + 0.5
    y_min, y_max = np.array(dataMat)[:, 1].min() - 0.5, np.array(dataMat)[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    # 预测网格中每个点的决策函数
    Z = np.array([predict_decision_function(oS, np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界（Z=0的等高线）
    ax.contour(xx, yy, Z, levels=[0], colors='blue', linewidths=2)
    # 绘制区域（contourf）
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['lightgreen', 'lightblue'], alpha=0.4)

    # 突出显示支持向量
    # oS.alphas 是一个矩阵，np.where 返回一个元组，我们需要第一个元素作为索引
    alphas_non_zeros_index = np.where(oS.alphas.A > 0)[0]
    for i in alphas_non_zeros_index:
        # dataMat 是一个列表的列表，转换为numpy数组以便索引
        circle = Circle((np.array(dataMat)[i, 0], np.array(dataMat)[i, 1]), 0.05, facecolor='none', edgecolor=(0, 0, 0), linewidth=3,
                        alpha=0.5)
        ax.add_patch(circle)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.title("支持向量 (Support Vectors)")
    plt.show()

if __name__ == "__main__":
    # 从指定文件加载数据
    dataMatIn, classLabels = loadData("Data/svm2.txt")

    # 运行SMO算法以获取偏置项(b)、alpha乘子和OptStruct对象
    # C是正则化参数，toler是KKT条件违反的容忍度，maxIter是最大迭代次数
    # kTup指定核类型和参数，例如 ('rbf', 1.3) 表示sigma=1.3的RBF核
    b, alphas, oS = smoPro(dataMatIn, classLabels, 200, 0.001, 40, ('rbf', 1.3))

    print("\n--- 计算结果 (Calculation Results) ---")
    print("b (偏置项):", b)
    # 打印非零的alpha值（对应于支持向量）
    print("支持向量的Alphas (Alphas for Support Vectors):\n", alphas[alphas > 0])

    # 绘制结果
    plotPoint(dataMatIn, classLabels)
    # 将OptStruct对象 (oS) 传递给绘图函数，用于基于核的绘图
    plotDesignSurface(np.asmatrix(dataMatIn), classLabels, oS)
    plotSupportVector(np.asmatrix(dataMatIn), classLabels, oS)
