import numpy as np

class LogisticRegression:
    '''
    将线性模型的结果用对数几率+阈值输出，转为离散变量，实现了回归到分类的连接。
    模型：sigmod函数：
    预测目标属于伯努利分布--伯努利分布同正态分布一样属于指数分布族--指数分布族的3个假设--推导出sigmod表达式
    详见：http://logos.name/archives/187
    策略：损失函数：
    由极大似然函数取对数再取反所得。
    首先要统一概率表达式：P(y|x,θ)=hθ(x)^y*(1−hθ(x))^(1−y)
    然后极大似然函数是P的连乘，损失函数是其对数取反。
    算法：梯度下降：
    利用g'(z) = g(z)(1-g(z))  z=w*x
    求导即得梯度为 X(hθ(X)−Y)
    故θ=θ−αX(hθ(X)−Y)
    '''
    def __init__(self,X,y,lr=0.01):
        self.X = np.c_[np.ones(len(X)),X]
        self.y = y
        self.theta = np.ones(self.X.shape[1])
        self.lr = lr
        self.sigmod = lambda X: 1/(1+np.e**(-X.dot(self.theta)))

    def loss(self):
        '''
        J(θ)=−lnL(θ)=−∑(y(i)log(hθ(x(i)))+(1−y(i))log(1−hθ(x(i))))其中i=[1,m]
        '''
        id1 = np.where(self.y==1)[0]
        id0 = np.where(self.y==0)[0]
        loss1 = -np.log(self.sigmod(self.X[id1])).sum()
        loss0 = -np.log(1-self.sigmod(self.X[id0])).sum()
        return loss1+loss0

    def train(self,max_iter = 400):
        '''BGD'''
        self.lossrst = []
        while len(self.lossrst)<=max_iter:
            errors = self.sigmod(self.X)-self.y
            self.theta -= self.lr*self.X.T.dot(errors)
            missed = ((self.X.dot(self.theta)>0)!=self.y).sum()
            self.lossrst.append((self.loss(), missed, self.theta))
        return 'train done!'

if __name__=='__main__':
    
    import pandas as pd
    df = pd.read_csv('./StatlogHeart.csv',header=None)
    y = df.pop(13).values-1
#    from sklearn.datasets import load_iris
#    iris = load_iris()
#    df = pd.DataFrame(iris.data, columns=iris.feature_names).iloc[:100,:2]
#    y = iris.target[:100]
    
    df = df.transform(lambda x:(x-x.mean())/x.std()) ## 标准化不可少

    LR = LogisticRegression(df.values,y,lr=0.0008)
    LR.train()
    train_info = pd.DataFrame(LR.lossrst,columns=['loss','missed','theta'])
    train_info.plot(y=['loss','missed'],secondary_y=['missed'],marker='o')
