# -*- coding: utf-8 -*-
import numpy as np
'''
EM算法
对于含有隐变量的概率模型，极大化对数似然函数时，涉及到对数内加和的式子的求导，这样的式子没有解析解。
注意到琴生不等式:对于凹函数有f(E[x])≥E[f(x)]，
将对数似然函数稍做变换，将 对数内求期望 转换成 ≥先对数再求期望，而先对数再期望是好求导的。
就是找到一个当前参数θi下的方便求导的下界函数B，
再对此下界函数求极大时候参数的取值（这里可以推导出，对B求导等价于对李航书中的Q函数求导）θi+1
将θi+1作为参数重算B，再对B求极大得到θi+2
执行上述过程，随着B不断取得最大，似然函数也最大直到收敛。
根据上述可知，只能找到局部最大值。所以EM对初始值敏感，要多试不同初始值求最佳。
具体到代码上：
1、初始化参数
2、求出 待下一步函数B的导数为0更新参数时 所需要的量，不一定非要求出B，求出需要的量即可。
3、根据2中的量计算新θ，回到步骤2，继续。
4、当θ变化很小或者Q函数变化很小，认为已收敛，停止迭代。
'''
class EM:
    
    def __init__(self,y,w):
        '''EM是一大类方法，对应不同概率分布。本例对应李航书中例题的二次分布'''
        self.y = y
        self.w = w
        self.MLE = [0]
        
    def mle(self):
        '''极大似然做停止条件等价于Q函数'''
        π,p,q = self.w
        p0, p1 = π*(1-p)+(1-π)*(1-q), π*p+(1-π)*q
        _, [count0, count1] = np.unique(self.y,return_counts=True)
        return p0**count0 * p1**count1
    
    def e_step(self):
        '''Q函数可看做期望，所以叫E步骤。本函数根据Q函数计算M步骤求导会用到的部分。'''
        π,p,q = self.w
        μ = lambda y: π*p/(π*p+(1-π)*q) if y else π*(1-p)/(π*(1-p)+(1-π)*(1-q))
        return np.array(list(map(μ,self.y)))
    
    def m_step(self,μ):
        '''返回求导Q函数为0(取最大值，所以叫M步骤)时各参数解析解。由E步骤结果得出。'''
        return μ.mean(), μ.dot(self.y)/μ.sum(), self.y.dot(1-μ)/(1-μ).sum()
        
    def train(self,eps = 1e-5):
        self.MLE.append(self.mle())
        while self.MLE[-1]-self.MLE[-2]>eps:
            μ_new = self.e_step()
            self.w = self.m_step(μ_new)
            self.MLE.append(self.mle())
        return 'trian done!'

if __name__=='__main__':
    w = np.array([.4,.6,.7])
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    em = EM(y,w)
    em.train()
    print(em.w) ## 与李航例题结果相同
    print(em.MLE)


