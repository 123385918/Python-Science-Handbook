import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    '''
    将线性模型的结果用对数几率+阈值输出，转为离散变量，实现了回归到分类的连接。
    '''
    def __init__(self,X,y,lr=0.1):
        self.X = np.c_[np.ones(len(X)),X]
        self.theta = np.zeros(X.shape[1])
        self.l_rate = lr

    def classify(self,X,thresold=0.5):
        '''
        X: 2-D matirx
        output: 0 or 1
        '''
        rst = X.dot(self.theta)
        return (rst>0.5).astype(int)

#    def miss_class(self,X):
#        '''
#        output:list of missClassified's id
#        '''
#        y_hat = self.classify(X)
#        return np.where(y_hat!=self.y)[0]

    def train(self,min_miss_ratio = .4):
        '''
        BGD
        '''
        y_hat = self.classify(X=self.X)
        miss_class = np.where(y_hat!=self.y)[0]
        while miss_class.size>len(self.y)*min_miss_ratio:
            m_X,m_y = self.X[miss_class],self.y[miss_class]
            self.theta -= self.lr*(m_X*(y_hat[miss_class]-m_y)).sum(1)
            miss_class = self.miss_class()
        return self.theta