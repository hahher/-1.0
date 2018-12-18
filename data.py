import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

""" 读入数据"""
data_all = pd.read_csv('data_all.csv', encoding='gbk')

"""划分训练集，测试集"""
x = data_all.drop(['status'], axis=1)
y = data_all['status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)

"""逻辑回归"""
lg = LogisticRegression(random_state=2018)
lg.fit(x_train, y_train)

"""SVM"""
svc = svm.SVC(random_state=2018)
svc.fit(x_train, y_train)

"""决策树"""
dt = DecisionTreeClassifier(random_state=2018)
dt.fit(x_train, y_train)

"""输出结果"""
print(lg.score(x_test, y_test))
print(dt.score(x_test, y_test))
print(svc.score(x_test, y_test))
