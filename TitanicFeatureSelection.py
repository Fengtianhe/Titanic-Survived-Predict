# -*- coding: utf-8 -*-
# 获取特征贡献度
# %%
import pandas

titanic = pandas.read_csv("./train.csv")
# 看头三行数据，默认5
# print(titanic.head(3))
# 统计每一列数据特征 count(总计), mean(均值), std(方差), min(最小值), 25%, 50%, 75%, max
# print(titanic.describe())

# 缺失值的填充, 只能针对数字类型的进行填充，像年龄这样的，用平均值
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
# print(titanic['Age'])
# 打印Sex列去重后的值
# print(titanic['Sex'].unique())

# 定位到Sex列，值 == male的变为0，把字符串转换为机器能理解的数据
titanic.loc[titanic['Sex'] == "male", "Sex"] = 0
titanic.loc[titanic['Sex'] == "female", "Sex"] = 1
# print(titanic['Sex'].unique())

# 使用众数的来填充缺失值,因为众数可能有很多，所以取第一个
# print(titanic['Embarked'].mode())
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
# 替换数值
titanic.loc[titanic['Embarked'] == "S", "Embarked"] = 0
titanic.loc[titanic['Embarked'] == "C", "Embarked"] = 1
titanic.loc[titanic['Embarked'] == "Q", "Embarked"] = 2
# 一样的数据补齐操作
titanic['Parch'] = titanic['Parch'].fillna(titanic['Parch'].mode()[0])

# 利用feature_selection计算特征的重要程度
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

scores = -np.log(selector.pvalues_)
print(sum(scores))

# 绘图
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()