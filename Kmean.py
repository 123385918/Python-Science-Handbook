# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

dist = lambda X, centers : np.linalg.norm(X[None,:] - centers[:,None], axis=2) ## k*m
agg = lambda keys,func:pd.DataFrame(X).groupby(keys).agg(func).values


class Kmean:
    def __init__(self,X):
        self.X = X ## m*n
        
    def train(self,k,centers=None):
        
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
    X = np.array([[0, 2], [0, 0], [1, 0], [5, 0], [5, 2]])
    kmean = Kmean(X)
    kmean.train(k=2)
    kmean.losses
