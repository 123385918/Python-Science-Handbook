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

    def _g(self,i):
        '''
        g(i) = Σaj*yj*xj*xi+b
        '''
        return (self.alpha * self.y).dot(self.Gram[:,i]) + self.b

    def _kkt(self,i,g):
        '''判断以i为索引的样本点是否符合KKT条件'''
        if self.alpha[i]==0:
            return self.y[i]*g >= 1
        elif self.alpha[i]==self.C:
            return self.y[i]*g <= 1
        else:
            return self.y[i]*g == 1
        
    def train(self,C,tol=0.001):
        self.C = C
        self.tol = tol ## 迭代精度，当|a_new-a_old|<e,停止迭代
        self.alpha = np.zeros(len(self.y)) ## 初始设置全部样本点都是非支持向量
        self.Gram = self.X.dot(self.X.T) ## 普通时用gram矩阵，加核时换成核矩阵
        self.b = 0
        self.E = np.array([self._g(i)-self.y[i] for i in range(len(self.y))])

        while e > self.tol:

            ## 外层循环--第一个变量a1的选取
            ## 首先遍历0<a1<C的样本点(间隔边界上的支持向量)，若均符合KKT条件，
            ## 则继续找在a1=C的样本点(间隔边界内的支持向量+误分的支持向量)，若均符合KKT条件，
            ## 则继续找在a1=0的样本点(非支持向量)，若均符合KKT条件，此时alpha列表即所求。
            for i in self._reorder():
                
                g = self._g(i)
                if self._kkt(i,g)==False:
                    a1 = self.alpha[i]
                    E1 = self.E[i]
                    j = self.E.argmax() if E1<0 else self.E.argmin()
                    a2 = self.alpha[j]
                    E2 = self.E[j]
                    break
            else:
                self.w
                self.b
                return 'train done'
            
            # eta=K11+K22-2K12
            eta = self.Gram[i,i]+self.Gram[j,j]-2*self.Gram[i,j]

