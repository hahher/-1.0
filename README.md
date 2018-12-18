# 金融贷款逾期的模型实现
通过给定数据集预测贷款用户是否会逾期，采用逻辑回归，SVM，决策树三种模型实现，并给出评分（数据无预处理，模型无调参）

[数据集下载](https://pan.baidu.com/s/1dtHJiV6zMbf_fWPi-dZ95g)

### 导入需要的包

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
```

###  数据导入和划分训练集,测试集

训练集和测试集三七分， "status" 是结果标签：0表示未逾期，1表示逾期，随机种子2018，使用pandas.drop函数进行分割标签
```
""" 读入数据"""
data_all = pd.read_csv('data_all.csv', encoding='gbk')

"""划分训练集，测试集"""
x = data_all.drop(['status'], axis=1)
y = data_all['status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)
```

###  模型训练与评分
···
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
···

### 模型评分
0.7484232655921513

0.6846531184302733

0.7484232655921513
