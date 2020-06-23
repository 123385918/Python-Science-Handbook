# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
class Kmean:
    def __init__(self,X):
        self.X = X ## m*n
        self.dist = lambda centers: np.linalg.norm(self.X[None,:] - centers[:,None], axis=2) ## k*m
        self.agg = lambda keys,func:pd.DataFrame(self.X).groupby(keys).agg(func).values
        
    def train(self,k,centers=None):
        self.maxiter = 1000
        
        ## 初始化类中心
        if centers==None:
            centers = self.X[np.random.choice(self.X.shape[0],k, replace=False)]
        
        ## 训练类中心
        loss = []
        while len(loss) < self.maxiter:
            
            dist = self.dist(centers) ## k*m
            
            cluster_id = dist.argmin(axis=0) ## m*1
            
            centers = self.agg(cluster_id, np.mean)
            
            loss.append(self.agg(cluster_id, np.var))
            
        return cluster_id
    
    def predict(self,X):
