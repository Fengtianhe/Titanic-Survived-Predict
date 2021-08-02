# -*- coding: utf-8 -*-
# 训练模型导出
# %%
import pandas
import joblib
import numpy as np

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

alg = joblib.load('./RandomForest.pkl')
titanic = pandas.read_csv('test.csv')

# 缺失值的填充, 只能针对数字类型的进行填充，像年龄这样的，用平均值
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())
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

test_predictors = (titanic[predictors])
# 预测结果
test_predictions = alg.predict(test_predictors)

print("总人数"+ str(len(test_predictions)) + "预测生还人数" + str(np.sum(test_predictions == 1)))