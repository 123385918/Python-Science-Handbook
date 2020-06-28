# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
dist = lambda X, centers : np.linalg.norm(X[None,:] - centers[:,None], axis=2) ## k*m
agg = lambda keys,func:pd.DataFrame(X).groupby(keys).agg(func).values

'''
原理：
1、初始化选择k个质心，
2、根据质心求出X的分类（单样本距离哪个质心近，就属于哪一类），
3、根据2求出的分类重算质心（均值），
4、重复2,3，直到质心不变。
类中样本到其质心距离总和为损失函数。对初始值敏感，不保证全局最优。
'''

class Kmean:
    def __init__(self,X):
        self.X = X ## m*n
        
    def train(self,k,centers=None): ## centers:list,not np.ndarray
        
        ## 初始化类中心
        self.centers = np.array(centers or 
                    self.X[np.random.choice(self.X.shape[0],k, replace=False)])
        
        ## 训练类中心
        self.losses = []
        while True:
            cluster_id = self.predict(self.X)
            self.centers = agg(cluster_id, np.mean) ## k*m
            loss = agg(cluster_id, lambda x: len(x)*x.var(ddof=0)).sum().round(4)
            if self.losses and loss==self.losses[-1]: break
            self.losses.append(loss)
        return 'train done!'
        
    def predict(self, X):
        _dist = dist(X, self.centers) ## k*m
        cluster_id = _dist.argmin(axis=0) ## m*1
        return cluster_id
    
if __name__=='__main__':
    import matplotlib as mpl
    c_map = mpl.colors.ListedColormap('rgb')
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    def plot(y=iris.target,title='iris.target'):
        pd.DataFrame(iris.data,columns=iris.feature_names).plot.scatter(s=20,
                x='petal length (cm)',y='petal width (cm)',color=y,
                colormap=c_map,marker='o',colorbar=None,title = title)
    plot()
        
    kmean = Kmean(X)
    kmean.train(k=3)
    y_hat = kmean.predict(X)
    plot(y_hat,'y_hat')
