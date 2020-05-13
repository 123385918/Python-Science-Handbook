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
    
    def forward(self,O,record = False):
        α = self.π*self.B[:,O[0]]
        α_T = [α.tolist()]
        for t in range(1,len(O)):
            α = α.dot(self.A)*self.B[:,O[t]]
            if record: α_T.append(α.tolist())
        return np.array(α_T) if record else α.sum()
    
    def backward(self,O,record = False):
        β = np.ones_like(self.π,dtype=float)
        β_T = [β.tolist()]
        for t in range(len(O)-2,-1,-1):
            β = np.dot(self.A*self.B[:,O[t+1]],β)
            if record: β_T.append(β.tolist())
        return np.array(β_T[::-1]) if record else np.dot(self.π*self.B[:,O[0]],β)
    
    def em_fit(self,O,N,maxiter=50): ## O：观测序列 N：隐状态个数
        V = np.unique(O)
        self.A = np.ones([N,N])/N
        self.B = np.ones([N,len(V)])/len(V)
        self.π = np.random.sample(N)
        self.π /= self.π.sum()
        self.p = [0]
        T_V = (O[:,None]==V).astype(int) ## T行V列的one-hot矩阵
        while len(self.p)<=maxiter:
            ## e_step：求当前参数下使Q函数导数为0时，有用部分的值
            T_α = self.forward(O, record = True)
            T_β = self.backward(O, record = True)
            ## m_step：根据e_step得到的值，按照解析解更新Q函数参数
            T_αβ = T_α*T_β
            self.A *= T_α[:-1].T.dot(T_β[1:]*self.B[:,O[1:]].T)/T_αβ[:-1].sum(0)[:,None]
            self.B = T_αβ.T.dot(T_V) / T_αβ.sum(0)[:,None]
            self.π = T_αβ[0] / T_αβ[0].sum(0)
            ## 记录当前λ下的O的概率
            self.p.append(T_αβ[0].sum())
        return 'train done!'
    
    def dp_pred(self,O):
        '''dp数组定义:dp[t,i]定义为，t时的状态为i的1~t个状态的最大概率。
           递推条件：dp[t,i] = max(dp[t-1,:]*A[:,i])*B[i,O[t]]
        '''
        dp = np.zeros((len(O),self.N))
        dp[0] = self.π*self.B[:,O[0]]
        for i in range(1,len(O)):
            tmp = dp[i-1,:,None]*self.A
            dp[i-1] = np.argmax(tmp,axis=0) ## 记下Ψ
            dp[i] = np.max(tmp,axis=0)*self.B[:,O[i]]
        path = [dp[i].argmax()]
        for i in range(len(O)-2,-1,-1): ## 回溯
            path.append(int(dp[i,path[-1]]))
        return path[::-1], dp[-1].max()
    
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
    import pandas as pd
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
    O = np.array([0, 1, 0])
    hmm = HMM(A,B,π)
    ## 问题一--观测概率
    hmm.backward(O) ## 0.130218
    hmm.forward(O) ## 0.130218
    hmm.eval_prob_direct(O) ## 0.130218
    hmm.eval_prob_direct2(O) ## 0.13021800000000003
    ## 问题二--参数估计
    hmm.em_fit(O,N=2,maxiter=20)
    pd.DataFrame(hmm.p,columns=['P(O|λ)']).plot(y='P(O|λ)')
    ## 问题三--隐状态预测
    hmm = HMM(A,B,π)
    hmm.dp_pred(O) ## ([2, 2, 2], 0.01467) 与李航结果相同