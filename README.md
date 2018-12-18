# 金融贷款逾期的模型实现
通过给定数据集预测贷款用户是否会逾期，采用逻辑回归，SVM，决策树三种模型实现，并给出评分（数据无预处理，模型无调参）

[数据集下载](https://pan.baidu.com/s/1dtHJiV6zMbf_fWPi-dZ95g)

## 导入需要的包

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
```
