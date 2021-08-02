# -*- coding: utf-8 -*-
# 利用逻辑回归验证泰坦尼克生存数据
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

# 线性回归模型库
from sklearn.linear_model import LogisticRegression
# 用于训练模型衡量，交叉验证：
# 例如：
# 1、把数据集分为"训练集"和"验证集"，
# 2、把"训练数据集"分成3份，代号：A,B,C。用A,B训练，用C验证；用A,C训练，用B验证；用B,C训练，用A验证。平均得到训练结果
from sklearn.model_selection import cross_val_score

# 指定哪些特征用于机器学习
# 船舱等级、性别、年龄、随行兄弟姐妹，随行父母，船票价格，登船地点
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# 导入逻辑回归的模型
alg = LogisticRegression(random_state=1)

scores = cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)

print("准确率：" + str(round(scores.mean() * 100, 4)) + "%")
