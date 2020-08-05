# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:53:12 2020

@author: Administrator
"""
## 接受-拒绝采样

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


## 待采样分布
p_pdf = lambda x: norm.pdf(x,loc=30,scale=10)+norm.pdf(x,loc=80,scale=20)

## 建议分布
q_pdf = lambda x: norm.pdf(x,loc=50,scale=30)

## 求包络系数M
X = np.arange(-50,151)
X_p, X_q = p_pdf(X), q_pdf(X)
M = np.max(X_p/X_q)


## 接受-拒绝过程
def acc_rej(n=int(1e5)):
    
    ## 按建议分布q生成数据z
    z = norm.rvs(loc=50,scale=30,size=n)

    ## 生成数据z 的p(x) 和 q(x)
    p_z = p_pdf(z)
    q_z = q_pdf(z)
    u = np.random.uniform(low=0,high = M*q_z)
    
    ## 返回接受样本作为采样结果
    return z[u<=p_z]

s = acc_rej(int(1e7))


plt.figure()
plt.plot(X,X_p,label='p')
plt.plot(X,X_q*M,label='q')
sns.distplot(s)


size = int(1e7)
sigma = 1.2
z = np.random.normal(loc = 1.4,scale = sigma, size = size)
qz = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*(z-1.4)**2/sigma**2)
k = 2.5
#z = np.random.uniform(low = -4, high = 6, size = size)
#qz = 0.1
#k = 10
u = np.random.uniform(low = 0, high = k*qz, size = size)

pz =  0.3*np.exp(-(z-0.3)**2) + 0.7* np.exp(-(z-2.)**2/0.3)
sample = z[pz >= u]
plt.figure()
plt.hist(sample,bins=150, alpha=.3, normed=True, edgecolor='black')
plt.hist(z,bins=150, normed=True, alpha=.3, edgecolor='red')
plt.scatter(z[:10000],qz[:10000], color='red')
plt.show()

