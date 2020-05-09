# -*- coding: utf-8 -*-
import numpy as np
"""
HMM
问题1：概率计算方法:给定λ=(A,B,π)和观测序列O，求P(O|λ)
直接计算法：
按照概率公式，列举所有可能的长度为T的状态序列，求各状态序列与观测序列的联合概率，
对所有可能的状态序列的联合概率求和。这是一棵深度为T，各节点的子节点为所有隐藏状态的完整树。
可写成DFS的遍历或递归。
"""

class HMM:
    
    def __init__(self,A=None,B=None,π=None):
        self.A = A
        self.B = B
        self.π = π
        self.N = len(π) ## 隐藏态个数
    
    def forward(self,O):
        α = self.π*self.B[:,O[0]]
        for t in range(1,len(O)):
            α = α.dot(self.A)*self.B[:,O[t]]
        return α.sum()
    
    def backward(self,O):
        β = np.ones_like(self.π,dtype=float)
        for t in range(len(O)-2,-1,-1):
            β = np.dot(self.A*self.B[:,O[t+1]],β)
        return np.dot(self.π*self.B[:,O[0]],β)
    
    def eval_prob_direct(self,O):
        rst = 0
        for n in range(self.N):
            stack = [(self.π[n]*self.B[n,O[0]], 0, n)] ## [累计的概率，第1个观测i,当前状态s]
            while stack:
                p, i, s = stack.pop()
                if i ==len(O)-1:
                    rst += p
                    continue
                for nn in range(self.N):
                    stack.append((p*self.A[s,nn]*self.B[nn,O[i+1]], i+1, nn))
        return rst
    
    def eval_prob_direct2(self,O):
        self.rst = 0
        def epd(p,i,s):
            if i==len(O)-1:
                self.rst += p
                return
            [epd(p*self.A[s,n]*self.B[n,O[i+1]],i+1,n) for n in range(self.N)]
        [epd(self.π[n]*self.B[n,O[0]], 0, n) for n in range(self.N)]
        return self.rst
    
if __name__=='__main__':
    Q = [1, 2, 3]
    V = [1, 0] ## 0:红 1:白
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    π = np.array([0.2,
                  0.4,
                  0.4])
    # O = ['红', '白', '红']
    O = [0, 1, 0]
    hmm = HMM(A,B,π)
    hmm.backward(O) ## 0.130218
    hmm.forward(O) ## 0.130218
    hmm.eval_prob_direct(O) ## 0.130218
    hmm.eval_prob_direct2(O) ## 0.13021800000000003
