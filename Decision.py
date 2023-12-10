
def trainLinear(linear, x, y):
    #使用sklearn库的最小二乘估计训练一个线性模型
    linear.fit(x, y)
    return linear

def score(linear, x, y):
    #计算线性模型linear的精度
    right = 0
    p = linear.predict(x)
    for i in range(p.shape[0]):
        if p[i]<=0 and y[i]==-1 or p[i]>0 and y[i]==1:
            right += 1
    return right / x.shape[0]


class TreeNode:
    cato = None
    left_child = None
    right_child = None
    model = None
    def __init__(self, model=None, C=None, left=None, right=None):
        self.cato = C
        self.model = model
        self.left_child = left
        self.right_child = right

class DecisionTree:
    train_data = None
    test_data = None
    train_label = None
    test_data = None
    root = None
    def __init__(self):

