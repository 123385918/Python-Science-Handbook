# -*- coding: utf-8 -*-
import numpy as np

class SVC:
    def __init__(self,X,y,C):
        self.X = X
        self.y = y

    def _reorder(self):
        '''根据当前alpha和C，返回重排序后的index，用于第一次循环。
           重排序后的数组，0<a1<C的样本id在前，a1=C的样本id在中，a1=0的样本id在后。
        '''
        rst = np.r_[np.flatnonzero((self.alpha<self.C) & (self.alpha>0)),
                    np.flatnonzero(self.alpha==self.C),
                    np.flatnonzero(self.alpha==0)]
        return rst

    def _kkt(self,i):
        '''判断以i为索引的样本点是否符合KKT条件'''
        
    def train(self,C,tol):
        self.C = C
        self.tol = tol
        self.alpha = np.zeros(len(self.y)) ## 初始设置全部样本点都是非支持向量
        self.Gram = self.X.dot(self.X.T) ## 普通时用gram矩阵，加核时换成核矩阵

        ## 外层循环--第一个变量a1的选取
        ## 首先遍历0<a1<C的样本点(间隔边界上的支持向量)，若均符合KKT条件，
        ## 则继续找在a1=C的样本点(间隔边界内的支持向量+误分的支持向量)，若均符合KKT条件，
        ## 则继续找在a1=0的样本点(非支持向量)，若均符合KKT条件，此时alpha列表即所求。
        ids = self._reorder()
        for i in ids:
            

