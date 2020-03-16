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
    def __init__(self,dim,ids,val,label = None):
        self.val = val ## 分割值
        self.dim = dim ## 分割维度
        self.ids = ids ## 输入此node的样本id
        self.is_val = None
        self.not_val = None
        self.label = label
        self.num = 1 ## 用于后续统计该节点为根节点的子树有多少个节点
        self.gt = None
        self.miss_len = 0 ## 用于后续统计该节为根节点的子树有多少个误分样本
    def __str__(self):
        return '<d:{},v:{},n:{},is:{},not:{}>'.format(
                self.dim,self.val,self.num,self.is_val,self.not_val)

class CART_classify:

    def __init__(self,X,y,n=1,g=0):
        self.X = X
        self.y = y
        ## 停止分类条件
        self.min_num_in_leaf = n
        self.min_gini_in_dim = g
        ## 剪枝子树序列
        self.t_seq = []
        self.alpha = 0

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
            if self.max_gini_gain(ids)[-1]<=self.min_gini_in_dim:
                return
            node = Node(dim,ids,val)
            node.is_val = _build(ids[self.X[ids,dim]==val])
            node.not_val = _build(ids[self.X[ids,dim]!=val])
            if node.is_val==node.not_val==None:
                node.label = Counter(self.y[ids]).most_common()[0]
            else: ## 生成完全二叉树，子节点只可能都None或都有值
                node.num += node.is_val.num+node.not_val.num
            return node
            
        self.tree = _build(range(len(self.X)))
        return 'train done!'

    def _miss_len_and_gt(self,node):
        '''
        计算node树全部节点的g(T)值，返回新树及最小的g(t)即alpha
        '''
        if node.label!=None: ## 叶子节点，不可算gt
            node.miss_len = sum(self.y[node.ids] != node.label)
            return
        else: ## 非叶子节点，可算gt
            ## 将node子树简化为叶子节点时的误分
            counter = Counter(self.y[node.ids])
            Rt = 1-counter.most_common()[1]/len(node.ids)
            RT = node.miss
            node.miss_len = self._miss_len(node.is_val) + self._miss_l
            node.not_val.miss_len = self._miss_len(node.not_val)
        
    def _g_t(self,node):
        '''
        计算node树全部节点的g(T)值，返回新树及最小的g(t)即alpha
        '''
        stack = [node]
        while stack:
            i = stack.pop()
            if i.is_val and i.not_val:
                ## 将node子树简化为叶子节点时的误分
                counter = Counter(self.y[node.ids])
                i.gt = 1-counter.most_common()[1]/len(node.ids)
        ## node子树的误分
        RT = self._g_t(node.is_val)+self._g_t(node.not_val)
        node.gt = Rt-RT
        return node

    def _cut(self,alpha,node):
        '''
        将g(t)等于alpha的子树剪掉
        '''
        pass

    def prune(self,node):
        if node.is_val==node.not_val==None:
            return
        ## 计算更新node树中每个节点的误分类点个数
        
        ## 根据误分类点个数计算每个节点的g(t)，返回最小g(t)为alpha
        alpha,node = self._g_t(node)
        ## 删除最小g(t)的节点
        node = self._cut(alpha,node)
        ## 将结果树存入序列
        self.t_seq.append((alpha,node))
        ## 对结果树继续剪枝操作
        self.prune(alpha,node)
        
    def predict(self,x):
        i = self.tree
        while i.label==None:
            if x[i.dim]==i.val:
                i = i.is_val
            else:
                i = i.not_val
        return i.label