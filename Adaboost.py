# -*- coding: utf-8 -*-
import numpy as np
'''
前提：
在概率近似正确的框架下，一个概念是强可学习的充要条件是此概念是弱可学习。
所以，只要发现了弱学习算法，就能提升成强学习算法。
Adaboost思想是将前一轮错分的样本权重提高，学习一个本轮的弱学习器，与之前的弱学习器加权相加，
直到分类错误率可接受或学习器数量达限。其中的加权相加，是把分错率多的权重变小，少的权重变大。

Adaboost是模型为加法模型，学习算法为前向分步学习算法，损失函数为指数函数的分类问题。
加法模型：
fx = Σαi*gix
fm+1 = fm + α*g, α，g是第m+1轮的值
损失函数
L = e^(-y*f(x))
其中em是分类误差率
算法迭代
初始化训练集权重w，用w权重下的训练集求出最佳基分类器g及其分类误差率e（w加权）。
将fm+1 = fm + α*g代入L，将g用训练数据选好代入，L对α求导使其等于0，求得α=1/2*ln(1/e-1)
fm+1 = fm + α*g，α和g均为上述所求，得到fm+1
更新w = w*np.exp(-α*y*y_hat)用于下次求最佳基分类器。
依次迭代直到分错率达标或学习器数量达限。
'''
class Node:
    def __init__(self,v,d):
        self.val = v ## 分割值
        self.dim = d ## 分割维度
        self.left = None ## 小于等于放左边
        self.right = None ## 大于放右边

class BaseCART:

    def __init__(self,max_depth=1):
        self.max_depth = max_depth

    def gini_y(self,sub_y,sub_w):
        if len(sub_y)<1:return 0
        id1 = np.where(sub_y==1)[0]
        p1 = sub_w[id1].sum()/sub_w.sum()
        return 2*p1*(1-p1)

    def gini(self,sub_X,sub_y,sub_w):
        rst = (None,None,np.inf) ##(分割维度，分割值，最小基尼系数)
        for col in range(sub_X.shape[1]):
            unique = np.sort(np.unique(sub_X[:,col]))
            if len(unique)==1:
                return (None,None,0)
            for j in (unique[1:]+unique[:-1])/2:
                id1 = np.where(sub_X[:,col]>j)[0]
                id2 = np.where(sub_X[:,col]<=j)[0]
                Gini_y_x = \
                (sub_w[id1].sum()/sub_w.sum())*self.gini_y(sub_y[id1],sub_w[id1])+\
                (sub_w[id2].sum()/sub_w.sum())*self.gini_y(sub_y[id2],sub_w[id2])
                rst = min(rst,(col,j,Gini_y_x),key = lambda x:x[-1])
        return rst

    def train(self,X,y,w):
        def bulid(ids,level):
            dim,val,gini = self.gini(X[ids],y[ids],w[ids])
            if (level==self.max_depth) or (gini==0):
                label = zip(*np.unique(y[ids],return_counts=True))
                return max(label,key=lambda x:x[1])[0]
            else:
                node = Node(val,dim)
                node.left = bulid(np.where(X[:,dim]<=val)[0],level+1)
                node.right = bulid(np.where(X[:,dim]>val)[0],level+1)
                return node
        return bulid(np.arange(len(y)),0)


class AdaBoost:
    def __init__(self, X, y, error = 0,n = 50):
        self.X = X
        self.y = y
        self.error = error ## 误判个数
        self.n = n ## 最大分类器个数

    def train(self, max_depth = 2):
        self.w = np.repeat(1/len(self.y),len(self.y))
        self.G = []
        self.alpha = []
        error = len(self.y)
        while error>self.error and len(self.G)<=self.n:
            G = BaseCART(max_depth).train(self.X,self.y,self.w)
            y_hat = np.array([self.pred0(G,xi) for xi in self.X])
            e = self.w[y_hat!=self.y].sum()
            alpha = 0.5 * np.log(1/e-1)
            self.alpha.append(alpha)
            self.G.append(G)
            error = (np.array([self.pred(x) for x in self.X])!=self.y).sum()
            print(error) ## 打印当前组合函数判错个数
            self.w *= np.exp(-alpha*self.y*y_hat)
            self.w /= self.w.sum()
        return 'train done!'

    def pred0(self,tree,x):
        i = tree
        while isinstance(i,Node):
            i = i.left if x[i.dim]<=i.val else i.right
        return i

    def pred(self,x):
        y_hat = np.array([self.pred0(tree,x) for tree in self.G])
        return np.sign(y_hat.dot(self.alpha))


if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names).iloc[:100,:2]
    y = np.where(iris.target[:100]>0,-1.0,1.0)
    # train model
    ab = AdaBoost(df.values,y)
    ab.train()
    # visialize
    pred = np.array([ab.pred(x) for x in ab.X])
    df.plot.scatter(x='sepal length (cm)',y='sepal width (cm)',c=(y==pred),
                    cmap='Spectral',title='alpha:%s'%ab.alpha,colorbar=False)
    # 李航
    X = np.arange(10).reshape(-1,1)
    y = np.repeat([1,-1,1,-1],[3,3,3,1])
    # train model
    ab = AdaBoost(X,y)
    ab.train(max_depth=1)
    print(ab.alpha) ## 与李航例题结果相同
