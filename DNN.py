# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:03:24 2020

@author: Administrator
"""
import numpy as np
sigmod = lambda x: 1/(1+np.exp(-x))


class DNN:
    
    def __init__(self,X,y,sizes):
        self.sizes = sizes
        self.W = [np.random.rand(w2,w1) for w1,w2 in zip(sizes[:-1],sizes[1:])]
        self.B = [np.random.rand(b) for b in sizes[1:]]
        self.L = len(sizes)

        
    def train(self,X,Y,epoch,alpha):
        '''SGD'''
        for t in range(epoch):
            i = np.random.randint(len(Y))
            x, y = X[i].copy(), Y[i]
            
            ##　前向激活求中间值
            F = []
            for w,b in zip(self.W, self.B):
                x = sigmod(w.dot(x)+b)
                F.append(x)
                
            ## 后向求误差值
            delta = [(x-y)*(x*(1-x))]
            ### W:[1:][::-1], B[1:][::-1], F[:-1][::-1]
            for w,b,f in zip(self.W[:-self.L:-1],self.B[:-self.L:-1],F[-2::-1]):
                tmp_delta = w.T.dot(delta[-1])*(f*(1-f))
                delta.append(tmp_delta)
                
            ## 前向更新参数
            for w,b in zip(self.W, self.B):
                tmp_delta = delta.pop()
                w -= alpha*(tmp_delta.dot())