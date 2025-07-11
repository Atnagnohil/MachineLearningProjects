# _*_ coding: utf-8 _*_
'''
时间:      2025/7/11 20:59
@author:  andinm
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
# 读取照片
def loadPhoto():
    img = mpimg.imread('Data/jjq.jpg')
    return img
# 数据处理 改变维度 特征归一化
def processData(img):
    img = img.reshape(img.shape[0], -1)
    # print(img.shape)
    img = img / 255.0
    return img
# 模型训练
def trainModel(img, K):
    model = PCA(n_components=K) # 需要将数据压缩到K维
    imgZip = model.fit_transform(img)
    # print(imgZip.shape)
    # print(f"贡献比{np.sum(model.explained_variance_ratio_):0.3%}")
    return imgZip, model
# 数据还原
def recoverData(imgZip, model, shape):
    imgRec = model.inverse_transform(imgZip).reshape(shape)
    print(imgRec.shape)
    return imgRec
# 展示处理之前和处理之后的图片
def plotPhoto(img, imgRec):
    plt.subplot(121), plt.title("Orign"), plt.imshow(img), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.title("Zip"), plt.imshow(imgRec), plt.xticks([]), plt.yticks([])
    plt.suptitle("Photo")
    plt.show()
if __name__ == "__main__":
    img = loadPhoto()
    # print(img.shape) # (800, 600, 3)
    imgProcess = processData(img)   # imgProcess是数据处理之后的图片
    imgZip, model = trainModel(imgProcess, 50) # imgZip是图片压缩到50维度的图片
    print(f"主成分个数 = {model.n_components}")
    print(f"贡献比 = {np.sum(model.explained_variance_ratio_):0.3%}")
    print(f"特征的方差 = {np.sum(model.explained_variance_):0.4f}")
    imgRec = recoverData(imgZip, model, img.shape)
    # print(imgRec[imgRec < 0])
    # print(imgRec[imgRec > 1])
    '''imgRec存在小于0大于1的情况'''
    imgRec[imgRec<0] = 0+1e-4
    imgRec[imgRec>1] = 1-1e-4
    plotPhoto(img, imgRec)



