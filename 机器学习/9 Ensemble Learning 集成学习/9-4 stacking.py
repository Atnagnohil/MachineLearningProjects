# _*_ coding: utf-8 _*_
'''
时间:      2025/7/18 18:42
@author:  andinm
'''
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np




def loadData(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = data[:, 0:1]
    return X, y

def featureNormalize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), train_size=0.8, random_state=0)
    return X_train, X_test, y_train, y_test


def tarinModel(X_train, X_test, y_train, y_test):
    clf1 = KNeighborsClassifier(n_neighbors=5)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=lr,use_probas=True)
    for model in [clf1,clf2,clf3,lr,sclf]:
        model.fit(X_train,y_train)
        y_test_hat = model.predict(X_test)
        print(model.__class__.__name__,',test accuarcy:',accuracy_score(y_test,y_test_hat))

if __name__ == "__main__":
    XOrigin, y = loadData('Data/wine.data')
    X = featureNormalize(XOrigin)
    X_train, X_test, y_train, y_test = splitData(X, y)
    tarinModel(X_train, X_test, y_train, y_test)
'''
KNeighborsClassifier ,test accuarcy: 0.9722222222222222
RandomForestClassifier ,test accuarcy: 1.0
GaussianNB ,test accuarcy: 0.9166666666666666
LogisticRegression ,test accuarcy: 1.0
StackingClassifier ,test accuarcy: 0.9444444444444444
'''