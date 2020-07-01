# -*- coding: utf-8 -*-
import numpy as np

def pca(X, k):
    '''X: m*n,m个样本，每个样本有n个特征
       k: 降到k维
    '''
    m, n = X.shape
    if k>n: return X
    X = X - X.mean(axis=0) ## 中心化
    cov = np.cov(X,rowvar=False,ddof=m-1) ## 协方差矩阵系数为1
    val, vec = np.linalg.eig(cov) ## 特征值+特征向量
    vec = vec[:,np.argpartition(-val,k-(k==n))[:k]] ## 取前k个最大特征值对应的特征向量
    return X.dot(vec) ## 返回降维后的X

if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    X = np.array([(2.5,2.4), (0.5,0.7), (2.2,2.9), (1.9,2.2), (3.1,3.0), 
                  (2.3, 2.7), (2, 1.6), (1, 1.1), (1.5, 1.6), (1.1, 0.9)])
    X1 = pca(X,1)
    X2 = pca(X,2)
    plt.scatter(X[:,0],X[:,1], marker='o',c='red')
    plt.scatter(X1,np.zeros(len(X1)), marker='o',c='blue')
    plt.scatter(X2[:,0],X2[:,1], marker='*',c='blue')
