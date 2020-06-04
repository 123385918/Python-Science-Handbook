# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:34:50 2020

@author: Administrator
"""

import numpy as np

# =============================================================================
# 损失函数导数定义
# =============================================================================
der_mse = lambda y_hat,y: y_hat - y
der_llh = lambda y_hat,y: y ## 必须接softmax激活函数，否则错误

class SoftLayer:
    
    def __init__(self):
        pass
    
    def forward(self,X,record = False):
        rst = np.exp(X)/np.exp(X).sum(1,keepdims=True)
        if record: self.temp = rst
        return rst
    
    def backward(self, cum_grad):
        return self.temp-cum_grad ## 必须接der_llh损失函数导数，否则错误
    
    def update(self, l_rate):
        pass

class LinearLayer:
    
    def __init__(self, size_in: int, size_out: int):
        self.W = np.random.rand(size_in, size_out) ## X*W+B
        self.B = np.random.rand(1, size_out)
        
    def forward(self,X,record=False):
        if record: self.temp = X
        return X.dot(self.W) + self.B
    
    def backward(self,cum_grad):
        self.grad_W = np.matmul(self.temp.T,cum_grad)
        self.grad_B = np.matmul(cum_grad.T, np.ones(len(self.temp)) )
        return np.matmul(cum_grad,self.W.T)
    
    def update(self, l_rate):
        self.W -= self.grad_W * l_rate/(len(self.temp))
        self.B -= self.grad_B * l_rate/(len(self.temp))
        
class SigmodLayer:
    
    def __init__(self):
        pass
    
    def forward(self,X,record = False):
        rst = 1/(1+np.exp(-X))
        if record: self.temp = rst
        return rst
    
    def backward(self, cum_grad):
        return self.temp*(1-self.temp)*cum_grad
    
    def update(self, l_rate):
        pass
    
class ReluLayer:
    
    def __init__(self):
        pass
    
    def forward(self,X,record = False):
        rst = np.where(X < 0, 0, X)
        if record: self.temp = rst
        return rst
    
    def backward(self, cum_grad):
        return np.where(self.temp > 0, 1, 0) * cum_grad
    
    def update(self, l_rate):
        pass
    
class DNN:
    def __init__(self,layers:list):
        self.layers = layers
        
    def predict(self,X,record=False):
        for layer in self.layers:
            X = layer.forward(X, record=record)
        return X.argmax(1)
    
    def train(self,X,Y,testX,testY,loss=der_mse,batch=10,epoch=50,alpha=.1):
        '''batch-GD'''
        self.info = []
        for t in range(epoch):
            batches = np.split(np.random.permutation(len(X)),
                               np.arange(len(X),step=batch)[1:])
            for ids in batches:
                
                ## 前向传播激活，记录求导时用到的输入或输出值
                forward = X[ids].copy()
                for layer in self.layers:
                    forward = layer.forward(forward, record=True)
                    
                ## 反向传播梯度，计算各层参数梯度
                grads = loss(forward, Y[ids]) ## 损失函数MSE导数y_hat-y
                for layer in self.layers[::-1]:
                    grads = layer.backward(grads)
                    
                ## 根据梯度更新各层参数
                for layer in self.layers:
                    layer.update(alpha)
                        
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
    iris.target = pd.get_dummies(iris.target,dtype=float).values

    X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=.3,random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # train model
    ## 最普通的sigmod激活+mse损失函数
    layers = [LinearLayer(4,8),SigmodLayer(),LinearLayer(8,3),
              SigmodLayer()]
    dnn = DNN(layers)
    dnn.train(X_train,y_train,X_test,y_test,
              loss=der_mse,batch=10,epoch=50,alpha=1)
    info = pd.DataFrame(dnn.info)
    info.plot(x='t',y='right',marker='o',ms=3)
    
    ## 对分类问题，softmax激活+对数似然损失函数效果更好
    layers = [LinearLayer(4,8),ReluLayer(),LinearLayer(8,3),
              SoftLayer()]
    dnn = DNN(layers)
    dnn.train(X_train,y_train,X_test,y_test,
              loss=der_llh,batch=10,epoch=20,alpha=.1)
    info = pd.DataFrame(dnn.info)
    info.plot(x='t',y='right',marker='o',ms=3)
    
#    x=np.array([[1,2]],dtype=float)
#    y=np.array([[1]],dtype=float)
#    layers[0].W=np.array([[2,1,3],[1,3,2]],dtype=float)
#    layers[0].W=np.array([[2,1,3],[1,3,2]],dtype=float)
#    layers[0].B=np.array([[1,2,3]],dtype=float)
#    layers[2].W=np.array([[1,3],[1,3],[2,2]],dtype=float)
#    layers[2].B=np.array([[2,1]],dtype=float)
#    layers[4].W=np.array([[1],[3]],dtype=float)
#    layers[4].B=np.array([[2]],dtype=float)
#    dnn = DNN(layers)
#    
#    ## 前向
#    o0 = x.copy()
#    a0 = layers[0].forward(o0) ## linear
#    o1 = layers[1].forward(a0) ## sigmod
#    a1 = layers[2].forward(o1) ## linear
#    o2 = layers[3].forward(a1) ## sigmod
#    a2 = layers[4].forward(o2) ## linear
#    o3 = layers[5].forward(a2) ## sigmod
#    y_hat = o3.copy()
#    ## 后向
#    y_hat_der = y_hat-y
#    o3_der = y_hat_der*o3*(1-o3)
#    a2_der_W = np.matmul(o2.T,o3_der)
#    a2_der_B = np.matmul(o3_der.T, np.ones(len(o2)) ) ## o3_der.sum(),得到单元素值
#    a2_der = np.matmul(o3_der,layers[4].W.T)
#    o2_der = a2_der*o2*(1-o2)
#    a1_der_W = np.matmul(o1.T, o2_der)
#    a1_der_B = np.matmul(o2_der.T, np.ones(len(o1)) )
#    a1_der = np.matmul(o2_der,layers[2].W.T)
#    o1_der = a1_der*o1*(1-o1)
#    a0_der_W = np.matmul(o0.T, o1_der)
#    a0_der_B = np.matmul(o1_der.T, np.ones(len(o0)) )
#    a0_der = np.matmul(o1_der,layers[0].W.T)