import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
from publicmodule import data_tf_yb

DT = DecisionTreeClassifier(criterion="gini")

train_data = np.array(pd.read_excel("./imgs/Train/traits.xlsx",index_col=None))
train_label = np.array(pd.read_excel("./imgs/Train/labels.xlsx",index_col=None))
test_data = np.array(pd.read_excel("./imgs/Test/traits.xlsx",index_col=None))
test_label = np.array(pd.read_excel("./imgs/Test/labels.xlsx",index_col=None))
print(test_data)
dstd = np.load("dstd.npy")
dmean = np.load("dmean.npy")
train_data = data_tf_yb(train_data,dmean,dstd)
test_data = data_tf_yb(test_data,dmean,dstd)




model = DT.fit(train_data,train_label)
p_ans = model.predict(test_data)
correct = 0


for i in range((test_data.shape)[0]):
    if(p_ans[i] == test_label[i]):
        correct += 1

print(1.0*correct / test_data.shape[0])
