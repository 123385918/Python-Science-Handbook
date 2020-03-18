# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import copy
"""
CART分类树
特征分类--评价指标
信息熵：https://blog.csdn.net/u014688145/article/details/53212112，详细推导H(X)=−∑pi*lnpi
条件熵：H(Y|X)=∑(x=xi的数据占全部x的比例）*（x=xi的子样本集的熵）
信息增益：H(Y)-H(Y|X)
基尼指数（及基尼增益）：
熵计算复杂些，所以对其用泰勒公式一阶近似得到基尼指数计算方法，
但基尼指数对应的是熵，而不是信息增益H(D)-H(D|A)。所以信息增益应该对应基尼增益G(D)-G(D|A),
这也是为何基尼指数G(D|A)要选小的，因为选小的G(D|A)才对应基尼增益最大，等同于信息增益最大。
同理，用信息增益选择也可以换成选择熵H(D|A)最小那个。
注：从以下代码看出，基尼增益和基尼指数的效果相同。

特征选择--离散特征：
每个维度的每个特征都作为二分点算基尼增益，求出全部基尼增益的最大那个即当前分割点、分割维。
但是后续选分割点还能选该维度。
特征选择--连续特征：
对连续特征的维度，将此维度的取值排序(假设有m个），取两两中间值得到m-1个，用这m-1个值作为划分点，
求出全部基尼增益的最大增益即当前分割点、分割维。但是后续选分割点还能选该维度。

缺失值处理--已选的特征A为判别特征，a样本的A特征有缺失：
A的取值分布：A1:3个, A2:5个, A3:2个. 于是a样本同时划入3个子树,
但A1子树中a样本权重3/10,A2子树中a样本权重5/10,A1子树中a样本权重2/10.
缺失值处理--要选择判别特征,其中待选项A特征有缺失：
100个样本,A特征有值的60个,无值的40个.则用40个计算特征选择参数(基尼,信息增益...),
结果乘以占比60/100作为最终结果,与其他特征比较后选择.

剪枝--后剪枝
计算每个节点的gt值，是（单节点误分类数-节点做根的子树的误分类数）/（节点做根的子树的节点数-1）
上面公式是分类误差率,也可以用基尼/信息增益计算.
每次剪gt值最小那个子树。得到完整树到根节点的序列.
在验证集上验证各子树，选最优。
详见 http://mlwiki.org/index.php/Cost-Complexity_Pruning

CART回归树
主要区别在于连续特征怎么划分,怎么预测.
连续特征划分:
在特征A的取值中,所有取值排序后,在两两数字中间任选一数使得特征A的数组被分为两半,求两半的方差之和.
在所有方差之和中选最小的划分即A特征的划分,再与其他特征划分选出最小,即当前的最佳划分
预测方式:
分类树输出投票最多那个,回归树输出均值或中位数.
"""
class Node:
    def __init__(self,dim,ids,val,label = None):
        self.val = val ## 分割值
        self.dim = dim ## 分割维度
        self.ids = ids ## 输入此node的样本id
        self.is_val = None
        self.not_val = None
        self.label = label
        self.leaf_num = 0 ## 该节点拥有的叶节点数目，用于后续剪枝
        self.gt = None ## 该节点gt值，用于后续剪枝，gt最小的点先剪掉

    def __str__(self):
        if self.label:
            return '<label:{},counter:{}>'.format(self.label,self.ids)
        elif self.gt:
            return '<d:{},v:{},leaf_n:{},gt:{},is:{},not:{}>'.format(
                self.dim,self.val,self.leaf_num,self.gt,self.is_val,self.not_val)
        else:
            return '<d:{},v:{},leaf_n:{},is:{},not:{}>'.format(
                self.dim,self.val,self.leaf_num,self.is_val,self.not_val)

class CART_classify:

    def __init__(self,X,y,n=1,g=0):
        self.X = X
        self.y = y
        ## 停止分类条件
        self.min_num_in_leaf = n
        self.min_gini_in_dim = g
        ## 剪枝子树序列
        self.alpha = np.inf

    def _Gini_y_with_x(self, ids):
        if len(ids)==0:return 0
        _,count_y = np.unique(self.y[ids],return_counts=True)
        return 1-((count_y/len(ids))**2).sum()

    def max_gini_gain(self,ids):
        '''
        description: 输入样本ids，返回此切片的最大gini增益及其分割信息。
        dim: scalar
        return: tuple(分割维度，分割值，最大增益)
        '''
        rst = (None,None,-np.inf) ##(分割维度，分割值，最大增益)
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
                node = Node(dim,ids,val)
                node.leaf_num = 1
                node.label = Counter(self.y[ids]).most_common()[0][0]
                return node
            node = Node(dim,ids,val)
            node.is_val = _build(ids[self.X[ids,dim]==val])
            node.not_val = _build(ids[self.X[ids,dim]!=val])
            node.leaf_num += node.is_val.leaf_num+node.not_val.leaf_num
            return node
            
        self.tree = _build(np.arange(len(self.X)))
        return 'train done!'

    def _g_t(self,node):
        '''
        计算返回node树误分类个数,最小alpha。更新树中各节点gt值。
        http://mlwiki.org/index.php/Cost-Complexity_Pruning
        '''
        if node.label!=None: ## 叶子节点
            return sum(self.y[node.ids] != node.label), np.inf ## 返回误分类点数，gt最小值
        else: ## 非叶子节点
            ## 将node子树简化为叶子节点时的误分
            counter = Counter(self.y[node.ids])
            Rt = len(node.ids)-counter.most_common()[0][1]
            ## node子树的误分
            left_miss,left_alpha = self._g_t(node.is_val)
            right_miss,right_alpha = self._g_t(node.not_val)
            RT = left_miss+right_miss
            node.gt = (Rt-RT)/(node.leaf_num-1)/len(self.y)
            return RT,min(left_alpha,right_alpha,node.gt) ## 返回误分类点数,gt最小值

    def _cut(self,node):
        '''
        将g(t)等于alpha的子树剪掉
        '''
        if node.gt==self.alpha:
            label = Counter(self.y[node.ids]).most_common()[0][0]
            node = Node(dim=None,ids=node.ids,val=None,label = label)
            node.leaf_num = 1
            return node
        node.is_val = node.is_val and self._cut(node.is_val)
        node.not_val = node.not_val and self._cut(node.not_val)
        node.leaf_num = node.is_val.leaf_num+node.not_val.leaf_num if node.label==None else 1
        return node

    def prune(self):
        self.t_seq = [0]
        stack = self.tree and [self.tree]
        while stack:
            i = stack.pop()
            if i.label==None:
                self.alpha = self._g_t(i)[1]
                self.t_seq.append(i)
                self.t_seq.append(self.alpha)
                stack.append(self._cut(copy.copy(i)))
        self.t_seq.append(i) ## 加上最后的单节点树
        return 'prune done!'

    def predict(self,x):
        i = self.tree
        while i.label==None:
            if x[i.dim]==i.val:
                i = i.is_val
            else:
                i = i.not_val
        return i.label

if __name__=='__main__':
    import pandas as pd
    ## 李航例题
    df = pd.DataFrame(data = 
              [[1, 2, 2, 3, '否'],
               [1, 2, 2, 2, '否'],
               [1, 1, 2, 2, '是'],
               [1, 1, 1, 3, '是'],
               [1, 2, 2, 3, '否'],
               [2, 2, 2, 3, '否'],
               [2, 2, 2, 2, '否'],
               [2, 1, 1, 2, '是'],
               [2, 2, 1, 1, '是'],
               [2, 2, 1, 1, '是'],
               [3, 2, 1, 1, '是'],
               [3, 2, 1, 2, '是'],
               [3, 1, 2, 2, '是'],
               [3, 1, 2, 1, '是'],
               [3, 2, 2, 3, '否']],
    columns = ['A1', 'A2', 'A3', 'A4', 'LABEL'])
    y = df.pop('LABEL')

    ## FOR PRUNE
    'https://docs.google.com/document/d/1d0Mh6XBX9NVyDExkVRNFIbrGGIwDugNDnP75xuDX9Aw/pub'
#    from CART import CART_classify
    validation = pd.DataFrame(data = 
          np.c_[list('PPFFPPFF'),list('PFFPPFPP'),list('GBGBGBGB')],
    columns = ['T2', 'T3','LABEL'])
    validation = validation.loc[np.repeat(validation.index.to_numpy(),[40,0,20,1,10,4,5,20])]
    validation_y = validation.pop('LABEL')
    dtree2 = CART_classify(validation.values,validation_y.values)
    dtree2.train()
    print(dtree2.tree)
    dtree2.prune()
    # 当0<=α<0.04,第1个树最佳，
    # 当0.04<=α<0.08,第2个树最佳，
    # 当0.08<=α,第3个树最佳，
    # [0, <CART.Node object at 0x0000007791904EF0>,
    #  0.04, <CART.Node object at 0x0000007791910828>,
    #  0.08, <CART.Node object at 0x0000007791910BE0>]

