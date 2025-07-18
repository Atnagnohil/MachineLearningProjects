# _*_ coding: utf-8 _*_
'''
时间:      2025/7/18 18:10
@author:  andinm
'''
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import numpy as np
import time
import joblib
import warnings

# --- 全局设置 ---
# 设置matplotlib以正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 忽略XGBoost的用户警告
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')


def load_and_prepare_data():
    """
    加载LibSVM格式的数据并进行预处理。

    Returns:
        tuple: 包含训练集和测试集的特征与标签 (X_train, y_train, X_test, y_test)。
    """
    print("正在加载数据...")
    # 假设 'Data' 文件夹与此脚本位于同一目录下
    # 如果不是，请提供绝对路径
    try:
        # 使用 __file__ 获取当前脚本的绝对路径，更可靠
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'Data')
        train_path = os.path.join(data_dir, 'agaricus.txt.train')
        test_path = os.path.join(data_dir, 'agaricus.txt.test')

        # 使用 sklearn 加载 LibSVM 文件
        X_train, y_train = load_svmlight_file(train_path)
        X_test, y_test = load_svmlight_file(test_path)
    except FileNotFoundError:
        print("错误: 未找到数据文件。请确保 'agaricus.txt.train' 和 'agaricus.txt.test' 文件位于 'Data' 子目录中。")
        # 作为备用，创建一个虚拟的空目录
        if not os.path.exists('Data'):
            os.makedirs('Data')
        print("请将数据文件放入 'Data' 文件夹后重试。")
        exit()  # 找不到文件则退出程序

    # 将稀疏矩阵转换为稠密矩阵以便后续处理
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    print("数据加载完成。")
    return X_train, y_train, X_test, y_test


def perform_grid_search(X_train, y_train):
    """
    使用GridSearchCV执行网格搜索以找到最佳超参数。

    Args:
        X_train (np.array): 训练集特征。
        y_train (np.array): 训练集标签。

    Returns:
        tuple: 包含最佳估计器和最佳参数字典 (best_estimator, best_params)。
    """
    # 创建XGBoost分类器
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False  # 推荐设置以避免警告
    )

    # 定义参数网格 - 可以根据需要调整范围
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5],
        'gamma': [0, 0.1],
        'subsample': [0.8, 1.0]
    }

    print("\n开始网格搜索...")
    start_time = time.time()

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,  # 3折交叉验证
        n_jobs=-1,  # 使用所有可用的CPU核心
        verbose=1
    )

    # 执行网格搜索
    grid_search.fit(X_train, y_train)

    end_time = time.time()
    print(f"网格搜索完成! 用时: {end_time - start_time:.2f}秒")

    # 打印最佳参数和分数
    print("\n网格搜索找到的最佳参数组合:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"  对应的交叉验证准确率: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def train_with_early_stopping(X_train, y_train, X_val, y_val, best_params):
    """
    使用早停策略训练最终模型，并记录学习过程。

    Args:
        X_train (np.array): 训练集特征。
        y_train (np.array): 训练集标签。
        X_val (np.array): 验证集特征。
        y_val (np.array): 验证集标签。
        best_params (dict): 网格搜索找到的最佳参数。

    Returns:
        xgb.XGBClassifier: 训练好的最终模型。
    """
    print("\n使用最佳参数和早停机制开始训练最终模型...")

    # 复制一份最佳参数，避免修改原始字典
    final_params = best_params.copy()

    # 为了充分利用早停，设置一个较大的 n_estimators
    # 早停机制会自动在验证集性能不再提升时停止训练
    final_params['n_estimators'] = 1000

    # **核心修正**: 将 early_stopping_rounds 作为模型初始化参数传入
    final_params['early_stopping_rounds'] = 10

    # 添加其他固定参数
    final_params.update({
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': ['logloss', 'error']
    })

    # 创建最终模型，此时已包含早停参数
    model = xgb.XGBClassifier(**final_params)

    # eval_set 必须提供，以触发早停机制
    eval_set = [(X_train, y_train), (X_val, y_val)]

    # **核心修正**: model.fit() 中不再需要 early_stopping_rounds 参数
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False  # 设置为False以保持输出整洁，训练过程会记录在 evals_result_ 中
    )

    print("最终模型训练完成。")
    return model


def evaluate_model(model, X_test, y_test):
    """
    在测试集上评估模型性能。

    Args:
        model (xgb.XGBClassifier): 训练好的模型。
        X_test (np.array): 测试集特征。
        y_test (np.array): 测试集标签。

    Returns:
        float: 模型在测试集上的准确率。
    """
    print("\n正在评估模型在测试集上的性能...")
    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")

    return accuracy


def plot_learning_curves(model):
    """
    绘制学习曲线（训练集 vs 验证集的损失和错误率）。

    Args:
        model (xgb.XGBClassifier): 包含 evals_result_ 的已训练模型。
    """
    print("\n正在绘制学习曲线...")
    if not hasattr(model, 'evals_result_') or model.evals_result() is None:
        print("警告: 模型中没有找到 'evals_result_'，无法绘制学习曲线。")
        return

    results = model.evals_result()

    # 检查是否同时存在训练集和验证集的评估结果
    if 'validation_0' not in results or 'validation_1' not in results:
        print("警告: 缺少训练集或验证集的评估结果，无法进行对比。")
        return

    # 获取训练过程中的迭代次数
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    # 1. 绘制对数损失 (LogLoss) 曲线对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(x_axis, results['validation_0']['logloss'], label='训练集 LogLoss')
    ax1.plot(x_axis, results['validation_1']['logloss'], label='验证集 LogLoss')
    ax1.legend()
    ax1.set_ylabel('对数损失 (LogLoss)')
    ax1.set_title('XGBoost 学习曲线: 对数损失')
    ax1.grid(True)

    # 2. 绘制分类错误率 (Error) 曲线对比
    ax2.plot(x_axis, results['validation_0']['error'], label='训练集错误率')
    ax2.plot(x_axis, results['validation_1']['error'], label='验证集错误率')
    ax2.legend()
    ax2.set_xlabel('迭代次数 (Boosting Round)')
    ax2.set_ylabel('分类错误率')
    ax2.set_title('XGBoost 学习曲线: 分类错误率')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("学习曲线图已保存为 'learning_curves.png'")
    plt.show()


def plot_feature_importance(model, max_features=20):
    """
    绘制特征重要性图。

    Args:
        model (xgb.XGBClassifier): 训练好的模型。
        max_features (int): 要显示的最重要特征的数量。
    """
    print("\n正在绘制特征重要性图...")
    # 获取特征重要性
    feature_importances = model.feature_importances_

    # 按重要性降序排序
    indices = np.argsort(feature_importances)[::-1]

    # 只显示前N个最重要的特征
    top_n = min(max_features, len(feature_importances))

    # 创建特征名称列表
    feature_names = [f'特征 {i}' for i in indices[:top_n]]

    plt.figure(figsize=(12, 8))
    plt.title(f"Top {top_n} 特征重要性")
    plt.bar(range(top_n), feature_importances[indices][:top_n], align="center")
    plt.xticks(range(top_n), feature_names, rotation=45, ha='right')
    plt.xlim([-1, top_n])
    plt.ylabel("重要性得分 (F-score)")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("特征重要性图已保存为 'feature_importance.png'")
    plt.show()


if __name__ == '__main__':
    # 1. 加载数据
    X, y, X_test, y_test = load_and_prepare_data()

    # 2. 从原始训练数据中划分出一部分作为训练集，一部分作为验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify确保验证集和训练集标签分布一致
    )
    print(f"\n数据划分完成: 训练集({X_train.shape}), 验证集({X_val.shape}), 测试集({X_test.shape})")

    # 3. 在训练集上执行网格搜索找到最佳参数
    best_model_from_grid, best_params = perform_grid_search(X_train, y_train)

    # 4. 使用找到的最佳参数和早停机制，在(训练集+验证集)上训练最终模型
    final_model = train_with_early_stopping(
        X_train, y_train, X_val, y_val, best_params
    )

    # 打印早停信息
    # 注意: best_iteration 属性在 scikit-learn 接口中可能不直接可用
    # 实际的迭代次数可以通过 evals_result 的长度来确定
    actual_rounds = len(final_model.evals_result()['validation_0']['logloss'])
    print(f"早停机制在第 {actual_rounds} 轮后完成训练。")

    # 5. 在独立的测试集上评估最终模型
    accuracy = evaluate_model(final_model, X_test, y_test)

    # 6. 绘制最终模型的学习曲线
    plot_learning_curves(final_model)

    # 7. 绘制最终模型的特征重要性
    plot_feature_importance(final_model)

    # 8. 保存最终模型
    joblib.dump(final_model, 'best_xgboost_model.pkl')
    print("\n最终模型已保存为 'best_xgboost_model.pkl'")

    # 9. 保存最佳参数
    try:
        with open('best_params.txt', 'w', encoding='utf-8') as f:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
        print("最佳参数已保存为 'best_params.txt'")
    except IOError as e:
        print(f"保存参数文件失败: {e}")

'''
D:\Anaconda\Anaconda\envs\MachineLearning\python.exe "D:\python\PycharmProjects\PyProject2\机器学习\9 Ensemble Learning 集成学习\9-3-3-2 XGBoost代码实现2.py" 
正在加载数据...
数据加载完成。

数据划分完成: 训练集((5210, 126)), 验证集((1303, 126)), 测试集((1611, 126))

开始网格搜索...
Fitting 3 folds for each of 16 candidates, totalling 48 fits
网格搜索完成! 用时: 7.26秒

网格搜索找到的最佳参数组合:
  gamma: 0
  max_depth: 3
  n_estimators: 100
  subsample: 0.8
  对应的交叉验证准确率: 1.0000

使用最佳参数和早停机制开始训练最终模型...
最终模型训练完成。
早停机制在第 26 轮后完成训练。

正在评估模型在测试集上的性能...
测试集准确率: 1.0000

正在绘制学习曲线...
学习曲线图已保存为 'learning_curves.png'

正在绘制特征重要性图...
特征重要性图已保存为 'feature_importance.png'

最终模型已保存为 'best_xgboost_model.pkl'
最佳参数已保存为 'best_params.txt'

进程已结束，退出代码为 0

'''
