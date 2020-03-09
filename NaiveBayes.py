# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from collections import Counter
class NaiveBayes:
    def __init__(self,X,y,Px = None,Py = 'Multinormial'):
        self.X = X
        self.y = y
        self.f_map = ['gaussian' for _ in range(self.X.shape[1])] if Px is None else Px

    def multinomial(self,X,λ=1):
        '''
        laplace smoothing
        '''
        counter = Counter(X)
        return lambda x:(counter[x]+λ)/(len(X)+λ*len(counter.keys()))
    
    def bernoulli(self,X):
        pass
    
    def gaussian(self,X):
        mean,std = X.mean(),X.std
        return lambda x: (1/((2*np.pi)**.5*std))*np.exp(-(x-mean)**2/(2*std**2))

    def train(self):
        '''
        input:
            list of distribution name
        output:
            list of N+1 funcs
            N:dim of X+y
            funcs: input xi or yi, and output prediction
            e.g:{y1:[f_y,f_x1,f_x2,...,f_xn]}
        '''
        self.model = {}
        f_y = self.multinomial(self.y,λ=0)
        for i in np.unique(self.y):
            self.model[i] = [f_y]
            for dim,f_name in enumerate(self.f_map):
                f_tmp = super().__getattribute__(f_name)
                self.model[i].append(f_tmp(self.X[self.y==i,dim]))
        return 'train model success'

    def tran_predict(self,new_x):
        self.train()
        prob_list = {}

        for i,f_list in self.model.items():
            prob = f_list.pop(0)(i)
            for x_i,f_x in zip(new_x,f_list):
                prob*=f_x(x_i)
            prob_list[i]=prob
        return prob_list
