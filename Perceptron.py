import pandas as pd
import numpy as np

class Perceptron:
    '''
    感知机模型
    找到一个超平面θ∙x=0
    使得正例θ∙x>0，负例θ∙x<0
    
    感知机模型损失函数
    使误分类yθ∙x<0的所有样本，到超平面的距离之和最小。
    J(θ)=−∑ y(i)θ∙x(i)/||θ||
    由于不考虑最小值大小，只考虑最小值时θ取值，故简化为
    J(θ)=−∑ y(i)θ∙x(i)
    
    感知机模型损失函数的优化方法
    损失函数里面只有误分类的M集合里面的样本才能参与损失函数的优化。
    只能采用随机梯度下降（SGD）或者小批量梯度下降（MBGD）。
    关于θ的偏导数−∑ y(i)x(i)
    θ 的梯度下降迭代公式应该为：
　　θ=θ + α∑y(i)x(i)
    由于我们采用随机梯度下降，每次采用一个误分类的样本来计算梯度，
    假设采用第i个样本来更新梯度，则简化后的θ向量的梯度下降迭代公式为：
    θ=θ+αy(i)x(i)
    '''
    def __init__(self,data):
        self.theta = np.zeros(data.shape[1])
        self.l_rate = 0.1
        
    def sign(self,theta,x):
        rst = np.sign(theta.dot(x))
        return rst or 1
    
    def train(self,x,y):
        '''
        stochastic gradient descent
        '''
        false_nums=len(data)
        while false_nums:
            
        