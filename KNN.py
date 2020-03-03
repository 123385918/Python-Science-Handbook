import numpy as np

class Tree:
    def __init__(self,val,dim):
        self.val = val
        self.left = None
        self.right = None
        self.stard_dim = dim
class KDTree:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.access = np.ones_like(y)
    def bulid(self,acess):
        if self.X[acess].size==0:
            return None
        else:
            node = Tree(self.)

