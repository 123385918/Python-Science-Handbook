import numpy as np
'''
kd树本质是二叉搜索树，寻找最近邻的过程就是从起始叶子节点开始，对二叉搜索树进行有剪枝的中序遍历。其中：

中序遍历：新点可以看成是现有KD树的新叶子节点，它先访问父节点更新最近邻和最近距离，
再根据父节点的分割面判断是否访问兄弟节点，处理完兄弟节点后再向上访问父节点的父节点（就说成爷节点吧）
来更新最近邻和最近距离，根据爷节点的分割面判断是否访问爷节点的另一个子树，以此类推直到根节点。

剪枝：若点到当前最近邻的距离<点到当前节点的分割面的距离，则当前节点的另一枝不访问，类似剪枝。
'''
class Tree:
    def __init__(self,col,row):
        self.col,self.row = col,row
        self.left, self.right = None, None
        self.useful = True
def preorder(node):
    print(node.row)
    if node.left:
        preorder(node.left)
    if node.right:
        preorder(node.right)
def inorder(node):
    if node.left:
        inorder(node.left)
    print(node.row)
    if node.right:
        inorder(node.right)
class KDTree:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        ## 计算x到节点距离，计算x到节点的分割面距离
        self.x2i = lambda x,i:((x-self.X[i.row])**2).sum()
        self.x2s = lambda x,i:(x[i.col]-self.X[i.row,i.col])**2

    def build(self):
        def _bulid(ids):
            if len(ids):
                col = self.X[ids].std(0).argmax() ## 方差最大的那一维作为分割维
                ag = np.argsort(self.X[ids,col])
                row = ids[ag[len(ag)//2]]
                node = Tree(col,row)
                node.left = _bulid(ids[ag[:len(ag)//2]])
                node.right = _bulid(ids[ag[len(ag)//2+1:]])
                return node
        self.tree = _bulid(np.arange(len(self.X)))
        return 'success'

    def search(self,x):
        ## 找到对应叶子，生成待回溯路径
        i = self.tree
        path = []
        while i:
            path.append(i)
            #visited.add(i.row)
            i.useful = False
            i = i.left if x[i.col]<self.X[i.row,i.col] else i.right
        ## 自底向上，中序遍历求得最近邻
        NN = {'node':None,'dist':float('inf')}
        while path:
            i = path.pop()
            ## 更新点到点距离即超球体半径
            d_x2i = self.x2i(x,i)
            if d_x2i<=NN['dist']:
                NN['node'],NN['dist'] = i,d_x2i
            ## 判断超球体是否与分割面相交
            d_x2s = self.x2s(x,i)
            if d_x2s<=NN['dist']: ## 若相交则将兄弟节点为根节点的子树入栈
                if i.left and i.left.useful:
                    path.append(i.left)
                    i.left.useful = False
                if i.right and i.right.useful:
                    path.append(i.right)
                    i.right.useful = False
        return NN

if __name__=='__main__':

    import pandas as pd
    from sklearn.datasets import load_iris
    from matplotlib import cm
    import matplotlib.pyplot as plt
    cmap = cm.get_cmap('Spectral')
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names).iloc[:100,:2]
    df['label'] = np.where(iris.target[:100]>0,1,-1)
    # bulid kdtree
    kd = KDTree(df.iloc[:,:2].values,df.iloc[:,-1].values)
    kd.build()
    preorder(kd.tree)
    inorder(kd.tree)
    # search target
    p=kd.search(np.array([6.3,3.4]))
    kd.X[p['node'].row]
    # visialize
    df.plot.scatter(x='sepal length (cm)',y='sepal width (cm)',
                    c=df.label,cmap='Spectral')
    plt.scatter(6.3,3.4,color='g')
    circle1=plt.Circle((6.3,3.4), p['dist']**.5, color='y', fill=False)
    plt.gcf().gca().add_artist(circle1)
