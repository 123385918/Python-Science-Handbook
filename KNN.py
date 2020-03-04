import numpy as np

class Tree:
    def __init__(self,dim,val,label):
        self.dim,self.val,self.label = dim,val,label
        self.left = None
        self.right = None
def preorder(node):
    print(node.val)
    if node.left:
        preorder(node.left)
    if node.right:
        preorder(node.right)
def inorder(node):
    if node.left:
        inorder(node.left)
    print(node.val)
    if node.right:
        inorder(node.right)
class KDTree:
    def __init__(self,X,y):
        self.X = X
        self.y = y
    def _bulid(self,ids):
        if len(ids):
            dim = self.X[ids].std(0).argmax()
            ag = np.argsort(self.X[ids,dim])
            val = self.X[ids][ag[len(ag)//2]]
            label = self.y[ids][ag[len(ag)//2]]
            node = Tree(dim,val,label)
            node.left = self._bulid(ids[ag[:len(ag)//2]])
            node.right = self._bulid(ids[ag[len(ag)//2+1:]])
            return node
    def build(self):
        self.tree = self._bulid(np.arange(len(self.X)))
        return 'success'
    def search(self,x):
        pass


data = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
label = np.ones(len(data))
kd = KDTree(data,label)
tree1 = kd.bulid0(np.arange(len(kd.X)))
preorder(tree1)
inorder(tree1)
