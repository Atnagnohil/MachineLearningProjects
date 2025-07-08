# -*- coding: utf-8 -*-
'''
时间:         2025/7/7 21:54
@author:      andinm
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score



def loadData():
    """
    从'data/cluster_data.csv'加载数据。
    返回:
        np.ndarray: 加载的数据。
    """
    data = np.loadtxt('data/cluster_data.csv', delimiter=',')
    return data

def trainModel(X):
    """
    使用GridSearchCV训练AgglomerativeClustering模型以找到最佳参数。
    参数:
        X (np.ndarray): 用于聚类的输入数据。
    返回:
        tuple: 包含以下内容的元组：
            - labels (np.ndarray): 来自最佳模型的聚类标签。
            - best_score (float): 找到的最佳轮廓系数。
            - best_model (AgglomerativeClustering): 最佳拟合模型。
            - best_params (dict): 找到的最佳参数。
            - results (pd.DataFrame): 包含所有GridSearchCV结果的DataFrame。
    """
    # 定义参数网格（处理'ward'连接的特殊情况）
    param_grid = [
        # 组1：仅适用于'ward'连接的参数组合
        {
            'n_clusters': [2, 3, 4, 5],
            'linkage': ['ward'],  # 固定为'ward'
            'metric': ['euclidean']  # 'ward'仅适用于欧氏距离
        },
        # 组2：适用于其他连接方法的参数组合
        {
            'n_clusters': [2, 3, 4, 5],
            'linkage': ['complete', 'average', 'single'],  # 其他连接方法
            'metric': ['euclidean', 'cosine']  # 这些连接支持的度量
        }
    ]

    # 创建基础聚类模型
    base_model = AgglomerativeClustering()

    # 定义一个直接用于GridSearchCV的评分函数
    # 这个函数必须接受 estimator, X, 和 y_true (即使y_true对于无监督学习是None)
    def grid_scorer(estimator, X, y_true=None):
        """
        用于GridSearchCV的自定义评分函数，计算轮廓系数。
        参数:
            estimator: 聚类估计器。
            X (np.ndarray): 输入数据。
            y_true: 真实标签 (对于无监督学习通常为None，但GridSearchCV会传递)。
        返回:
            float: 轮廓系数，如果聚类无效则返回-1。
        """
        try:
            # 拟合估计器并预测标签
            labels = estimator.fit_predict(X)
            # 轮廓系数需要至少2个唯一的标签
            if len(np.unique(labels)) < 2:
                return -1 # 对于无效聚类返回一个低分
            return davies_bouldin_score(X, labels)
        except ValueError:
            # 处理参数可能不兼容或其他错误发生的情况
            return -1 # 发生错误时返回一个低分

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=grid_scorer, # 直接使用定义的评分函数
        cv=2, # 2折交叉验证
        refit=False, # 不自动重新拟合最佳估计器
        verbose=2, # 更详细的输出
        n_jobs=-1, # 使用所有可用的CPU核心
        error_score='raise' # 如果无法计算分数则引发错误
    )

    # 执行网格搜索
    grid_search.fit(X)

    # 获取并打印结果
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # 显示所有参数组合的结果
    results = pd.DataFrame(grid_search.cv_results_)

    # 使用找到的最佳参数重新拟合模型
    best_model = AgglomerativeClustering(**best_params)
    best_model.fit(X)

    return best_model.labels_, best_score, best_model, best_params, results

def draw(labels):
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cm_dark, s=20)
    plt.show()

if __name__ == "__main__":
    X = loadData()
    # print(X.shape)
    labels, best_score, best_model, best_params, results = trainModel(X)

    print(f"最佳参数: {best_params}")
    print(f"最佳轮廓系数: {best_score:.4f}")
    print("\n所有参数组合的评分:")
    print(results[['params', 'mean_test_score']].sort_values('mean_test_score', ascending=False))
    print("\n最佳模型聚类标签:")
    print(labels)
    draw(labels)
# # -*- coding: utf-8 -*-
# '''
# 时间:         2025/7/7 21:54
# @author:      andinm
# '''
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import silhouette_score
# import pandas as pd
#
#
# def loadData():
#     """
#     从'data/cluster_data.csv'加载数据。
#     返回:
#         np.ndarray: 加载的数据。
#     """
#     data = np.loadtxt('data/cluster_data.csv', delimiter=',')
#     return data
#
#
# def trainModel(X):
#     """
#     使用GridSearchCV训练AgglomerativeClustering模型以找到最佳参数。
#     参数:
#         X (np.ndarray): 用于聚类的输入数据。
#     返回:
#         tuple: 包含以下内容的元组：
#             - labels (np.ndarray): 来自最佳模型的聚类标签。
#             - best_score (float): 找到的最佳轮廓系数。
#             - best_model (AgglomerativeClustering): 最佳拟合模型。
#             - best_params (dict): 找到的最佳参数。
#             - results (pd.DataFrame): 包含所有GridSearchCV结果的DataFrame。
#     """
#     # 定义参数网格（处理'ward'连接的特殊情况）
#     param_grid = [
#         # 组1：仅适用于'ward'连接的参数组合
#         {
#             'n_clusters': [2, 3, 4, 5],
#             'linkage': ['ward'],  # 固定为'ward'
#             'metric': ['euclidean']  # 'ward'仅适用于欧氏距离
#         },
#         # 组2：适用于其他连接方法的参数组合
#         {
#             'n_clusters': [2, 3, 4, 5],
#             'linkage': ['complete', 'average', 'single'],  # 其他连接方法
#             'metric': ['euclidean', 'cosine']  # 这些连接支持的度量
#         }
#     ]
#
#     # 创建基础聚类模型
#     base_model = AgglomerativeClustering()
#
#     # 定义一个直接用于GridSearchCV的评分函数
#     # 这个函数必须接受 estimator, X, 和 y_true (即使y_true对于无监督学习是None)
#     def grid_scorer(estimator, X, y_true=None):
#         """
#         用于GridSearchCV的自定义评分函数，计算轮廓系数。
#         参数:
#             estimator: 聚类估计器。
#             X (np.ndarray): 输入数据。
#             y_true: 真实标签 (对于无监督学习通常为None，但GridSearchCV会传递)。
#         返回:
#             float: 轮廓系数，如果聚类无效则返回-1。
#         """
#         try:
#             # 拟合估计器并预测标签
#             labels = estimator.fit_predict(X)
#             # 轮廓系数需要至少2个唯一的标签
#             if len(np.unique(labels)) < 2:
#                 return -1  # 对于无效聚类返回一个低分
#             return silhouette_score(X, labels)
#         except ValueError:
#             # 处理参数可能不兼容或其他错误发生的情况
#             return -1  # 发生错误时返回一个低分
#
#     # 创建GridSearchCV对象
#     grid_search = GridSearchCV(
#         estimator=base_model,
#         param_grid=param_grid,
#         scoring=grid_scorer,  # 直接使用定义的评分函数
#         cv=2,  # 2折交叉验证
#         refit=False,  # 不自动重新拟合最佳估计器
#         verbose=2,  # 更详细的输出
#         n_jobs=-1,  # 使用所有可用的CPU核心
#         error_score='raise'  # 如果无法计算分数则引发错误
#     )
#
#     # 执行网格搜索
#     grid_search.fit(X)
#
#     # 获取并打印结果
#     best_params = grid_search.best_params_
#     best_score = grid_search.best_score_
#
#     # 显示所有参数组合的结果
#     results = pd.DataFrame(grid_search.cv_results_)
#
#     # 使用找到的最佳参数重新拟合模型
#     best_model = AgglomerativeClustering(**best_params)
#     best_model.fit(X)
#
#     return best_model.labels_, best_score, best_model, best_params, results
#
# def draw(labels, best_score, best_params, results):
#     # --- 可视化聚类结果 ---
#     mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
#     mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
#     # 绘制最佳聚类结果 (n_clusters = 2)
#     plt.figure(figsize=(12, 6))
#
#     # plt.subplot(1, 2, 1)
#     plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
#     plt.title(f'最佳聚类结果 (n_clusters={best_params["n_clusters"]})\n轮廓系数: {best_score:.4f}')
#     plt.xlabel('特征 1')
#     plt.ylabel('特征 2')
#     plt.grid(True)
#
#     # 绘制 n_clusters = 3 的聚类结果进行比较
#     # 查找 param_grid 中 n_clusters = 3 且 linkage 和 metric 与最佳参数相同的组合
#     # 如果最佳linkage是'ward'，则metric只能是'euclidean'
#     if best_params['linkage'] == 'ward':
#         params_3_clusters = {'n_clusters': 3, 'linkage': 'ward', 'metric': 'euclidean'}
#     else:
#         # 尝试使用最佳linkage和metric，但n_clusters为3
#         params_3_clusters = {'n_clusters': 3, 'linkage': best_params['linkage'], 'metric': best_params['metric']}
#
#     # 检查这个组合是否存在于results中，并获取其分数
#     score_3_clusters = -1  # 默认值
#     labels_3_clusters = None
#
#     # 遍历results DataFrame查找匹配的参数组合
#     for index, row in results.iterrows():
#         if row['params'] == params_3_clusters:
#             score_3_clusters = row['mean_test_score']
#             # 如果找到，则重新拟合模型以获取标签
#             model_3_clusters = AgglomerativeClustering(**params_3_clusters)
#             labels_3_clusters = model_3_clusters.fit_predict(X)
#             break
#
#     # 如果没有找到精确匹配，或者最佳linkage是'ward'但n_clusters=3时没有该组合，
#     # 则直接创建一个新的AgglomerativeClustering模型进行聚类
#     if labels_3_clusters is None:
#         # 尝试使用最佳linkage和metric，n_clusters=3
#         # 如果原始最佳参数是ward，metric必须是euclidean
#         if best_params['linkage'] == 'ward':
#             model_3_clusters_fallback = AgglomerativeClustering(n_clusters=3, linkage='ward', metric='euclidean')
#         else:
#             model_3_clusters_fallback = AgglomerativeClustering(n_clusters=3, linkage=best_params['linkage'],
#                                                                 metric=best_params['metric'])
#         labels_3_clusters = model_3_clusters_fallback.fit_predict(X)
#         # 重新计算轮廓系数，因为这个组合可能不在GridSearchCV的直接结果中
#         if len(np.unique(labels_3_clusters)) >= 2:
#             score_3_clusters = silhouette_score(X, labels_3_clusters)
#         else:
#             score_3_clusters = -1
#
#     plt.subplot(1, 2, 2)
#     plt.scatter(X[:, 0], X[:, 1], c=labels_3_clusters, cmap='viridis', s=50, alpha=0.8)
#     plt.title(f'聚类结果 (n_clusters=3)\n轮廓系数: {score_3_clusters:.4f}')
#     plt.xlabel('特征 1')
#     plt.ylabel('特征 2')
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.show()
#
# if __name__ == "__main__":
#     X = loadData()
#     labels, best_score, best_model, best_params, results = trainModel(X)
#
#     print(f"最佳参数: {best_params}")
#     print(f"最佳轮廓系数: {best_score:.4f}")
#     print("\n所有参数组合的评分:")
#     print(results[['params', 'mean_test_score']].sort_values('mean_test_score', ascending=False))
#     print("\n最佳模型聚类标签:")
#     print(labels)
#     draw(labels, best_score, best_params, results)


