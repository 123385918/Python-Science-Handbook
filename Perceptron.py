import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self,data,lr=0.1):
        self.theta = np.zeros(data.shape[1])
        self.l_rate = lr
        self.data = np.c_[np.ones(len(data)),data]
        
    def sign(self,x):
        rst = np.sign(self.theta.dot(x))
        return np.array([rst or 1])
    
    def train(self):
        '''
        stochastic gradient descent
        '''
#        miss_class = self._miss_class_id()
#        while miss_class:
#            miss_id = np.random.choice(miss_class,1)
#            X,y = np.split(self.data[miss_id],[self.data.shape[1]-1])
#
#            while self.sign(X).dot(y)<0:
#                self.theta += self.l_rate*X*y
#'''你好，我这里的意思就是每次都随机选择一个误分类的点做迭代。
#而你说的应该是每次迭代某一个误分类的点，一直梯度迭代到它不再被误分，再去迭代其他可能的误分类点。
#虽然都可以达到算法收敛，但个人倾向于我那种迭代方式，因为这样迭代没有偏向性，收敛会快一些。'''
#
#            miss_class = self._miss_class_id()
#        return self.theta
        miss_class = True
        while miss_class:
            permuted = np.random.permutation(len(self.data))
            miss_class=False
            for i in permuted:
                X, y = self.data[i,:-1],self.data[i,-1:]
                if self.sign(X).dot(y)<0:
                    self.theta += self.l_rate*X*y
                    miss_class=True
                    break
        return self.theta
    
class antithesisPerceptron:
    '''
    X: 2-D
    y: 1-D
    '''
    def __init__(self,X,y):
        # super().__init__(l_rate,data)
        self.l_rate = .1
        self.X = np.c_[np.ones_like(y,dtype=float),X]
        self.y = y
        self.beta = np.zeros_like(y,dtype=float)
        self.Gram = self.X.dot(self.X.T)
        
    def sign(self,x):
        '''
        x: scalar
        '''
        return np.sign(x) or -1
    
    
    def train(self):
        '''
        stochastic gradient descent with antithesis
        '''
        miss_class = True
        while miss_class:
            permuted = np.random.permutation(len(self.X))
            miss_class=False
            for i in permuted:
                y_hat = self.sign((self.beta * self.y).dot(self.Gram[:,i]))
                if self.y[i]*y_hat<0:
                    self.beta[i] += self.l_rate
                    miss_class = True
                    break
        return ((self.beta*self.y)[:,None]*self.X).sum(axis=0)


class BGDPerceptron:
    def __init__(self,X,y):
        self.lr = .1
        self.X = np.c_[np.ones_like(y),X]
        self.y = y
        self.theta = np.zeros(self.X.shape[1])
        
    def sign(self,x):
        return np.sign(x) or -1
    
    def miss_class(self):
        _y = np.sign((self.theta*self.X).sum(axis=1))
        y_y = np.where(_y<1,-1,1)*self.y
        return np.where(y_y<1)[0]
    
    def train(self):
        '''
        miss_class:list of missClassified's id
        '''
        miss_class = self.miss_class()
        while miss_class.size:
            m_X,m_y = self.X[miss_class],self.y[miss_class]
            self.theta += self.lr*(m_X*m_y[:,None]).sum(axis=0)
            miss_class = self.miss_class()
        return self.theta

if __name__=='__main__':
    
    from sklearn.datasets import load_iris
    from matplotlib import cm
    cmap = cm.get_cmap('Spectral')
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names).iloc[:100,:2]
    df['label'] = iris.target[:100]
    df.label.replace(0,-1,inplace=True)
    # train the model
    perceptron = Perceptron(df.values,1.0)
    a,b,c = perceptron.train() ## a+bx+cy=0
    # visialize
    df.plot.scatter(x='sepal length (cm)',y='sepal width (cm)',
                    c=df.label,cmap='Spectral')
    plt.plot(df['sepal length (cm)'],-(a+b*df['sepal length (cm)'])/c)
    
    # train the model2
    X = iris.data[:100,:2]
    y = np.where(iris.target[:100]<1,-1,1)
    
    p = antithesisPerceptron(X,y)
    a,b,c = p.train() ## a+bx+cy=0
    
    df = pd.DataFrame(np.c_[X,y],columns = iris.feature_names[:2]+['label'])
    df.plot.scatter(x='sepal length (cm)',y='sepal width (cm)',
                    c=df.label,cmap='Spectral')
    plt.plot(df['sepal length (cm)'],-(a+b*df['sepal length (cm)'])/c)
    
    # train the model3
    X = iris.data[:100,:2]
    y = np.where(iris.target[:100]<1,-1,1)
    
    p = BGDPerceptron(X,y)
    a,b,c = p.train() ## a+bx+cy=0
    
    df = pd.DataFrame(np.c_[X,y],columns = iris.feature_names[:2]+['label'])
    df.plot.scatter(x='sepal length (cm)',y='sepal width (cm)',
                    c=df.label,cmap='Spectral')
    plt.plot(df['sepal length (cm)'],-(a+b*df['sepal length (cm)'])/c)