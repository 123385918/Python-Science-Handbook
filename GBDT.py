# -*- coding: utf-8 -*-
import numpy as np
'''
GBDT也是集成学习Boosting家族的成员，但是却和传统的Adaboost有很大的不同。回顾下Adaboost，
我们是利用前一轮迭代弱学习器的误差率来更新训练集的权重，这样一轮轮的迭代下去。
GBDT也是迭代，使用了前向分布算法，但是弱学习器限定了只能使用CART回归树模型，同时迭代思路和Adaboost也有所不同。

在GBDT的迭代中，假设我们前一轮迭代得到的强学习器是ft−1(x), 损失函数是L(y,ft−1(x)), 
我们本轮迭代的目标是找到一个CART回归树模型的弱学习器ht(x)，让本轮的损失函数L(y,ft(x)=L(y,ft−1(x)+ht(x))最小。
也就是说，本轮迭代找到决策树，要让样本的损失尽量变得更小。

GBDT的思想可以用一个通俗的例子解释，假如有个人30岁，
我们首先用20岁去拟合，发现损失有10岁，这时我们用6岁去拟合剩下的损失，
发现差距还有4岁，第三轮我们用3岁拟合剩下的差距，差距就只有一岁了。
如果我们的迭代轮数还没有完，可以继续迭代下面，每一轮迭代，拟合的岁数误差都会减小。

从上面的例子看这个思想还是蛮简单的，但是有个问题是这个损失的拟合不好度量，
损失函数各种各样，怎么找到一种通用的拟合方法呢？
大牛Freidman利用最速下降法，提出了用损失函数的负梯度来拟合本轮损失的近似值，进而拟合一个CART回归树。
对于以MSE为损失函数的回归问题，负梯度就是残差。
对于以指数损失函数的分类问题，GBDT就是adaboost（参见adaboost损失函数求导）。
还有其他损失函数，详见刘建平相关博客和sklearn文档。
'''
class Node:
    def __init__(self,v,d):
        self.val = v ## 分割值
        self.dim = d ## 分割维度
        self.left = None ## 小于等于放左边
        self.right = None ## 大于放右边

        
class BaseRT:
    def __init__(self,max_depth=1):
        self.max_depth = max_depth

    def mse(self,sub_X,sub_y):
        rst = (0,0,np.inf) ##(分割维度，分割值，最小均方误差)
        for col in range(sub_X.shape[1]):
            unique = np.sort(np.unique(sub_X[:,col]))
            if len(unique)==1:
                return (None,None,0)
            for j in (unique[1:]+unique[:-1])/2:
                id1 = np.where(sub_X[:,col]<=j)[0]
                id2 = np.where(sub_X[:,col]>j)[0]
                MSE = sub_y[id1].var()*len(id1)+sub_y[id2].var()*len(id2)
                rst = min(rst,(col,j,MSE),key = lambda x:x[-1])
        return rst

    def train(self,X,y):
        def bulid(ids,level):
            if level==self.max_depth:
                return y[ids].mean() ## 当损失函数是MSE，c就是均值
            else:
                dim,val,mse = self.mse(X[ids],y[ids])
                node = Node(val,dim)
                node.left = bulid(np.where(X[:,dim]<=val)[0],level+1)
                node.right = bulid(np.where(X[:,dim]>val)[0],level+1)
                return node
        return bulid(np.arange(len(y)),0)


class GBDT:
    def __init__(self, X, y, min_loss = 0,n = 50):
        self.X = X
        self.y = y
        self.min_loss = min_loss ## 损失精度
        self.n = n ## 最大分类器个数

    def train(self, max_depth = 2):
        self.F = []
        residual = self.y.copy() ## 初始f0=0，残差是y-0=y
        loss = np.inf
        while loss>self.min_loss and len(self.F)<self.n:
            f = BaseRT(max_depth).train(self.X,residual)
            self.F.append(f)
            residual_hat = [self.pred0(f,xi) for xi in self.X]
            residual -= residual_hat
            y_hat = [self.pred(x) for x in self.X]
            loss = np.square(self.y-y_hat).sum()
            print(loss)
        return 'train done!'

    def pred0(self,tree,x):
        i = tree
        while isinstance(i,Node):
            i = i.left if x[i.dim]<=i.val else i.right
        return i

    def pred(self,x):
        y_hat_list = np.array([self.pred0(tree,x) for tree in self.F])
        return y_hat_list.sum()


if __name__ == '__main__':

    import numpy as np
    # 李航
    X = np.arange(1,11,dtype=float).reshape(-1,1)
    y = np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9.0,9.05])
    # train model
    gbdt = GBDT(X,y,n=6)
    gbdt.train(max_depth=1)
    print([x.val for x in gbdt.F]) ## 与李航例题结果相同
