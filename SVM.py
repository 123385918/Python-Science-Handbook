# -*- coding: utf-8 -*-
import numpy as np

class SVC:
    
    def __init__(self,X,y,C=1,tol=1e-3,eps = 5):
        self.X = X
        self.y = y
        self.C = C
        self.eps = eps ## alpha保留小数点后eps位，是更新的最小精度
        self.tol = tol ## 判断KKT条件的容错程度
        self.N = range(len(self.y))
        self.alpha = np.zeros_like(self.y,dtype = float) ## 初始设置全部样本点都是非支持向量
        self.Gram = self.X.dot(self.X.T) ## 普通时用gram矩阵，加核时换成核矩阵
        self.b,self.w = 0, 0
        self.E_cache = np.array([self.E(i) for i in self.N]) ## 存储预测偏差
        self.flag = True ## 数据集有违反KKT条件的标识
        
    def E(self,i):
        '''计算E(i) = g(i) - y(i)'''
        gi = (self.alpha * self.y).dot(self.Gram[:,i]) + self.b
        yi = self.y[i]
        return gi-yi
    
    def out_order(self):
        '''用于外部循环找到a1。根据当前alpha和C，返回重排序后的index。
           重排序后的数组，0<a1<C的样本id在前，a1=C的样本id在中，a1=0的样本id在后。
        '''
        rst = np.r_[np.random.permutation(np.flatnonzero((self.alpha<self.C) & (self.alpha>0))),
                    np.random.permutation(np.flatnonzero(self.alpha==self.C)),
                    np.random.permutation(np.flatnonzero(self.alpha==0))]
        return rst
    
    def in_order(self,E1):
        '''用于内部循环找到a2。把|E1-E2|最大的样本id放最前，
           再把alpha在(0,C)之间的样本id放中间，最后把训练集中其他样本id放后边
        '''
        rst = np.r_[np.flatnonzero((self.alpha<self.C) & (self.alpha>0)),
                    np.flatnonzero((self.alpha==self.C)|(self.alpha==0))]
        first = self.E_cache.argmin() if E1>0 else self.E_cache.argmax()
        firstID = np.flatnonzero(rst==first)[0]
        rst[firstID], rst[0] = rst[0], rst[firstID]
        return rst

    def train(self):
        while self.flag:
            for i in self.out_order():## 外层循环找a1
                '''判断a1，找到a2，优化a2，更新a1a2，更新b，更新E。仅当a1a2均成功更新才重新while'''
                ## 判断a1
                E1, a1, y1 = self.E_cache[i], self.alpha[i], self.y[i]
                self.flag = ((E1*y1<-self.tol) & (a1<self.C))|((E1*y1>self.tol) & (a1>0))
                if not self.flag: ## 若不违背KKT
                    continue
                ## 找到a2
                for j in self.in_order(E1):## 内层循环找a2
                    if j==i:
                        continue
                    E2, a2, y2 = self.E_cache[j], self.alpha[j], self.y[j]
                    ## 优化a2
                    L = max(0,a2-a1) if y1!=y2 else max(0,a1+a2-self.C)
                    H = min(self.C, self.C+a2-a1) if y1!=y2 else min(self.C, a1+a2)
                    if L==H:
                        print('L==H')
                        continue
                    eta = self.Gram[i,i]-2*self.Gram[i,j]+self.Gram[j,j]
                    if eta<=0:
                        print('eta<=0')
                        continue
                    a2_new = np.clip(a2 + y2*(E1-E2)/eta, L, H).round(self.eps)
                    ## 更新a1,a2
                    if a2_new==a2:
                        print('a2 changed too little')
                        continue
                    a1_new = np.round(a1+y1*y2*(a2-a2_new), self.eps)
                    self.alpha[i], self.alpha[j] = a1_new, a2_new
                    ## 更新b
                    b1_new = -E1-y1*self.Gram[i,i]*(a1_new-a1)-y2*self.Gram[j,i]*(a2_new-a2)+self.b
                    b2_new = -E2-y1*self.Gram[i,j]*(a1_new-a1)-y2*self.Gram[j,j]*(a2_new-a2)+self.b
                    if 0<a1_new<self.C:
                        self.b = b1_new
                    elif 0<a2_new<self.C:
                        self.b = b2_new
                    else:
                        self.b = (b1_new+b2_new)/2
                    ## 更新E
                    self.E_cache = np.array([self.E(i) for i in self.N])
                    break
                else:
                    continue
                break
        self.w = ((self.alpha*self.y)[:,None]*self.X).sum(0)
        return 'train done!'

if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names).iloc[:100,:2]
    y = np.where(iris.target[:100]>0,-1.0,1.0)
    # train model
    svc2 = SVC(df.values,y,C=100,tol=.001)
    svc2.train()
    # visialize
    w,b = svc2.w,svc2.b
    df.plot.scatter(x='sepal length (cm)',y='sepal width (cm)',c=y,cmap='Spectral',title = 'C=%s'%svc2.C)
    plt.plot(df.iloc[:,0],-(df.iloc[:,0]*w[0]+b)/w[1])
    for i in np.flatnonzero(svc2.alpha>0):
        plt.gcf().gca().add_artist(plt.Circle((df.iloc[i,0], df.iloc[i,1]), .05, color='y', fill=False))

    '''
        判断以i为索引的样本点是否符合KKT条件:
            ((E1*y1<-self.tol) & (a1<self.C))|((E1*y1>self.tol) & (a1>0))
        完全可以将这个self.tol换为0，我么就以换为0之后的条件来分析这个式子：
        self.y[i] * Ei < 0 and (self.alphas[i] < self.C)
        拆开左边的式子：Yi*(fxi-Yi)=Yi*fxi-1<0, 从而有Yi*fxi<1。
        此时根据KKT条件，我们应该取alpha_i = C，但是右边显示alpha_i < C,所以违背了KKT条件
        拆开右边的式子：Yi*(fxi-Yi)=Yi*fxi-1>0, 从而有Yi*fxi>1。
        此时根据KKT条件，我们应该取alpha_i = 0，但是右边显示alpha_i > C,所以违背了KKT条件
        因此，此判断式是找出了违背了KKT条件的alpha
        还有人问，为什么KKT条件有三个，此处只判断了两个？其实，此式确实判断了三个条件，只是合在了一起，下面是解释：
        注意，self.alphas[i] < self.C包含了0<alpha_i<C和alpha_i=0两个条件（同理另一个也包含了alpha_i=C的情况），
        所以alpha=0和alpha=C这两个KKT条件，被分别放在两个式子中判断了，
        0<alpha<C也被分成了两部分，这样三个条件就都有了判断
    '''

