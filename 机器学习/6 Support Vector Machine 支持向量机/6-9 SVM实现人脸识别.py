# _*_ coding: utf-8 _*_
'''
时间:      2025/7/5 21:58
@author:  andinm
'''
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def loadData():
    # 加载数据
    # 指定自定义下载路径
    custom_data_path = r"D:\python\PycharmProjects\PyProject2\机器学习\6 Support Vector Machine 支持向量机"  # 替换为你想要的路径

    # 加载数据并指定下载路径
    faces = fetch_lfw_people(
        min_faces_per_person=60,
        data_home=custom_data_path  # 关键参数：指定下载路径
    )
    print(faces.data.shape)
    print(faces.target_names)
    print(faces.images.shape)
    return faces

def plotPartData(faces):
    # 展示一部分数据
    fig, ax = plt.subplots(3, 5)
    fig.subplots_adjust(left=0.0625, right=1.2, wspace=1)
    for i, axi in enumerate(ax.flat):
        axi.imshow(faces.images[i], cmap='bone')
        axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
    plt.show()


def splitData(faces):
    ### 为了测试分类器的训练效果，将数据集分解成训练集和测试集进行交叉检验
    print("\n-----开始分割数据------")
    x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=42)
    return x_train, x_test, y_train, y_test


def trainModel(x_train, y_train):
    print("\n-----开始训练模型------")
    # 创建管道
    pca = PCA(n_components=150, whiten=True, random_state=42, svd_solver='randomized')
    '''将PCA的主成分数量也作为可调参数
        添加了svd_solver='randomized'以提高效率'''
    svc = SVC(kernel='rbf', class_weight='balanced', probability=True)
    '''class_weight='balanced'：处理类别不平衡问题
        probability=True：启用概率预测，便于后续分析'''
    model = make_pipeline(pca, svc)

    # 定义参数网格 - 使用更清晰的格式
    param_grid = {
        'svc__C':[1, 5, 10, 50],
        'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]  # 扩展gamma值范围
    }

    # 配置网格搜索 - 使用更规范的格式
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='accuracy',  # 主要评估指标
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1,  # 显示详细进度
        refit=True,  # 用最佳参数重新拟合模型      # 默认为True
        return_train_score=True  # 返回训练分数以供分析        # 默认为False
    )

    # 执行网格搜索
    grid_search.fit(x_train, y_train)

    # 输出最佳参数和分数
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 返回最佳模型
    return grid_search.best_estimator_

def predictData(x_test, y_test, model):
    print("\n-----开始预测数据------")
    # 预测数据
    y_fit = model.predict(x_test)
    # 进行比较
    fig, ax = plt.subplots(4, 6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(x_test[i].reshape(62, 47), cmap='bone')
        axi.set(xticks=[], yticks=[])
        axi.set_ylabel(faces.target_names[y_fit[i]].split()[-1],
                       color='black' if y_fit[i] == y_test[i] else 'red')
    fig.suptitle('预测错误的名字用红色标注', size=14)
    plt.show()
    return y_fit

### 打印分类效果报告，他会列举每个标签的统计结果，从而对评估器的性能有更全面的认识
# 绘制混淆矩阵
def modelAccess(y_test, y_fit):
    ### 打印分类效果报告，他会列举每个标签的统计结果，从而对评估器的性能有更全面的认识
    print("\n-----开始评估模型------")

    print(classification_report(y_test, y_fit, target_names=faces.target_names))

    # 绘制混淆矩阵
    mat = confusion_matrix(y_test, y_fit)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=faces.target_names,
                yticklabels=faces.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

if __name__ == "__main__":
    faces = loadData()
    plotPartData(faces)
    x_train, x_test, y_train, y_test = splitData(faces)
    model = trainModel(x_train, y_train)
    y_fit = predictData(x_test, y_test, model)
    modelAccess(y_test, y_fit)


'''
(1348, 2914)
['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush'
 'Gerhard Schroeder' 'Hugo Chavez' 'Junichiro Koizumi' 'Tony Blair']
(1348, 62, 47)

-----开始分割数据------

-----开始训练模型------
Fitting 5 folds for each of 16 candidates, totalling 80 fits
最佳参数: {'svc__C': 5, 'svc__gamma': 0.001}
最佳交叉验证分数: 0.8289

-----开始预测数据------

-----开始评估模型------
                   precision    recall  f1-score   support

     Ariel Sharon       0.65      0.87      0.74        15
     Colin Powell       0.83      0.88      0.86        68
  Donald Rumsfeld       0.70      0.84      0.76        31
    George W Bush       0.97      0.80      0.88       126
Gerhard Schroeder       0.76      0.83      0.79        23
      Hugo Chavez       0.93      0.70      0.80        20
Junichiro Koizumi       0.86      1.00      0.92        12
       Tony Blair       0.82      0.98      0.89        42

         accuracy                           0.85       337
        macro avg       0.82      0.86      0.83       337
     weighted avg       0.86      0.85      0.85       337


进程已结束，退出代码为 0

'''




