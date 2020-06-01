# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:03:24 2020

@author: Administrator
"""
import numpy as np
sigmod = lambda x: 1/(1+np.exp(-x))


class DNN:
    '''隐藏层均为sigmod激活函数
       sizes[5,4,2,1]:输入层5个元，隐藏层有2层，隐1有4个元，隐2有2个元，输出层1个元'''
    def initial(self,sizes):
        self.B = [np.random.rand(b) for b in sizes[1:]]
        self.W = [np.random.rand(w2,w1) for w1,w2 in zip(sizes[:-1],sizes[1:])] ## W:nrows:输出，ncols:输入
    
    def __init__(self,sizes):
        self.initial(sizes)
        self.sizes = sizes
    
    def predict(self,X):
        for w,b in zip(self.W, self.B):
            X = np.apply_along_axis(lambda x: sigmod(w.dot(x)+b), 1, X)
        return X.argmax(1)
    
    def train(self,X,Y,testX,testY,batch=10,epoch=50,alpha=.1):
        '''batch-GD'''
        self.info = []
        self.initial(self.sizes)
        for t in range(epoch):
            batches = np.split(np.random.permutation(len(X)),
                               np.arange(len(X),step=batch)[1:])
            for ids in batches:
                x, y = X[ids].copy(), Y[ids].copy()
                
                ## 前向激活求中间值
                F = [x]
                for w,b in zip(self.W, self.B):
                    x = np.apply_along_axis(lambda row: sigmod(w.dot(row)+b),1,x)
                    F.append(x)
                
                ## 后向求误差值
                δ = [(x-y)*(x*(1-x))]
                for w,f in zip(self.W[1:][::-1],F[1:-1][::-1]):
                    delta = np.apply_along_axis(lambda row: w.T.dot(row),1,δ[-1])
                    delta *= f*(1-f)
                    δ.append(delta)
                
                ## 前向更新参数
                δ.reverse()
                for w,b,d,f in zip(self.W, self.B, δ, F[:-1]):
                    grad_w = np.sum([i[:,None]*j for i,j in zip(d,f)],axis=0)
                    w -= alpha/batch * grad_w
                    b -= alpha/batch * d.sum(0)
                
            ## 记录训练信息
            Y_hat = self.predict(testX)
            self.info.append({'t':t,'right':(Y_hat==testY.argmax(1)).mean()})
        
        return 'train done!'

if __name__=='__main__':
    
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # load data
    iris = load_iris()
    iris.target = pd.get_dummies(iris.target).values

    X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=.3,random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # train model
    dnn = DNN(sizes=[4,5,4,3])
    dnn.train(X_train,y_train,X_test,y_test,batch=10,epoch=50,alpha=3)
    info = pd.DataFrame(dnn.info)
    info.plot(x='t',y='right',marker='o',ms=3)

    # load data
    mnist = fetch_openml('mnist_784', version=1, data_home='E:/Learn/algorithm_ljp')
    mnist.target = pd.get_dummies(mnist.target).values
    X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=.3,random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # train model
    dnn = DNN(sizes=[784,30,10])
    dnn.train(X_train,y_train,X_test,y_test,batch=10,epoch=30,alpha=10)
    info = pd.DataFrame(dnn.info)
    info.plot(x='t',y='right',marker='o',ms=3)