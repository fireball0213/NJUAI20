# -*- coding = utf-8 -*-
# Time : 2022/4/18 18:13
# Author : 201300086史浩男
# File : DT-sklearn.py
# Software : PyCharm

from sklearn import tree
from sklearn.tree import export_graphviz
X=[[1,0,1],[1,1,0],[0,0,0],[0,1,1],[0,1,0],[0,0,1],[1,0,0],[1,1,1]]
Y=[1,0,0,1,0,0,0,0]

clf=tree.DecisionTreeClassifier(criterion="entropy")
print(clf)
clf=clf.fit(X,Y)
print(clf)
print(clf.predict([[2,2,2]]))#预测属于哪个类
print(clf.predict_proba([[2,2,2]]))#预测属于每个类的概率
