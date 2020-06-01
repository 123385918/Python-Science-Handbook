# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:34:50 2020

@author: Administrator
"""

import numpy as np

class LinearLayer:
    
    def __init__(self, size_in: int, size_out: int):
        self.W = np.random.rand(size_in, size_out) ## X*W+B
        self.B = np.random.rand(1, size_out)
        
    def forward(self,X):
        '''input:N*size_in
           output: N*size_out'''
        return X.dot(self.W) + self.B
    
    def backward(self,X,cum_grad):
        self.grad_W = X.T.dot(cum_grad)
        self.grad_B = cum_grad.sum(1,keepdims=True)
        return cum_grad.dot(self.W.T)## ??
    
    def update(self,X,l_rate):
        self.W -= self.grad_W * l_rate/(len(X))
        self.B -= self.grad_B * l_rate/(len(X))
        
class SigmodLayer:
    
    def __init__(self):
        pass
    
    def forward(self,X):
        return 1/(1+np.exp(-X))
    
    def backward(self,X,cum_grad):
        f = self.forward(X)
        return f*(1-f)*cum_grad
    
class ReluLayer:
    
    def __init__(self):
        pass
    
    def forward(self,X):
        return np.where(X < 0, 0, X)
    
    def backward(self, X, cum_grad):
        return np.where(X > 0, 1, 0) * cum_grad
    
class DNN:
    def __init__(self,layers:list):
        self.layers = layers
        
    def predict(self,X):
        for layer in self.layers:
            X = layer.forward(X)
        return X.argmax(1)
    
    def train(self,X,Y,testX,testY,batch=10,epoch=50,alpha=.1):
        '''batch-GD'''
        self.info = []
        for t in range(epoch):
            batches = np.split(np.random.permutation(len(X)),
                               np.arange(len(X),step=batch)[1:])
            for ids in batches:
                x, y = X[ids].copy(), Y[ids].copy()
                
                ## 前向传播激活，获取各层输出
                output = [x]
                for layer in self.layers:
                    output.append(layer.forward(output[-1]))
                    
                ## 反向传播梯度，获取各层参数梯度
                grads = [output[-1]-y]
                for layer in self.layers[::-1]:
                    grads.append(layer.backward(x,grads[-1]))
                    
                ## 根据梯度更新各层参数
                for layer in self.layers:
                    if isinstance(layer,LinearLayer):
                        layer.update(x,alpha)
                        
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
    layers = [LinearLayer(2,3),SigmodLayer(),LinearLayer(3,2),
              SigmodLayer(),LinearLayer(2,1),SigmodLayer()]
    dnn = DNN(layers)
    dnn.train(X_train,y_train,X_test,y_test,batch=10,epoch=30,alpha=10)
    info = pd.DataFrame(dnn.info)
    info.plot(x='t',y='right',marker='o',ms=3)