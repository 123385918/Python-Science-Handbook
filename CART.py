# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
"""
CART分类树
特征选择--离散特征：
每个维度的每个特征都作为二分点算基尼增益，求出全部基尼增益的最大那个即当前分割点、分割维。
但是后续选分割点还能选该维度。
特征选择--连续特征：
对连续特征的维度，将此维度的取值排序(假设有m个），取两两中间值得到m-1个，用这m-1个值作为划分点，
求出全部基尼增益的最大增益即当前分割点、分割维。但是后续选分割点还能选该维度。
"""
class Node:
    def __init__(self,dim,val):
        self.val = val
        self.dim = dim
        self.is_val = None
        self.not_val = None
        self.label = None
    def __str__(self):
        return '<d:{},v:{},is:{},not:{}>'.format(
                self.dim,self.val,self.is_val,self.not_val)

class CART_classify:

    def __init__(self,X,y,n,g):
        self.X = X
        self.y = y
        ## 停止分类条件
        self.min_num_in_leaf = n
        self.min_gini_in_dim = g

    def _Gini_y_with_x(self, ids):
        if len(ids)==0:return 0
        _,count_y = np.unique(self.y[ids],return_counts=True)
        return 1-((count_y/len(ids))**2).sum()

    def max_gini_gain(self,ids):
        '''
        description: 输入样本ids，返回此切片的最大gini增益及其分割信息。
        dim: scalar
        ids: 1-D array
        return: tuple(分割维度，分割值，最大增益)
        '''
        rst = (None,None,-np.inf) ##(分割维度，分割值，最大增益)
        if len(ids) <= self.min_num_in_leaf:return rst
        Gini_y = self._Gini_y_with_x(ids = ids)
        for i in range(self.X.shape[1]):
            unique,inverse = np.unique(self.X[ids,i],return_inverse = True)
            for j in range(len(unique)):
    
                Gini_y_x = \
                (inverse==j).mean()*self._Gini_y_with_x(ids = ids[inverse==j])+\
                (inverse!=j).mean()*self._Gini_y_with_x(ids = ids[inverse!=j])
                rst = max(rst,(i,unique[j],Gini_y-Gini_y_x),key = lambda x:x[-1])

        return rst


    def train(self):
        def _build(ids):
            dim,val,gini_gain = self.max_gini_gain(ids)
            if self.max_gini_gain(ids)[-1]<=0:
                return
            node = Node(dim,val)
            node.is_val = _build(ids[self.X[ids,dim]==val])
            node.not_val = _build(ids[self.X[ids,dim]!=val])
            if node.is_val==node.not_val==None:
                node.label = Counter(self.y[ids]).most_common()[0]
            return node
        def _cut(node):
            pass
            
        self.tree = _build(range(len(self.X)))
        return 'train done!'

    def predict(self,x):
        i = self.tree
        while i.label==None:
            if x[i.dim]==i.val:
                i = i.is_val
            else:
                i = i.not_val
        return i.label