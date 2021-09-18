# -*- coding: utf-8 -*-
# 利用线性回归验证泰坦尼克生存数据
# %%
import pandas

titanic = pandas.read_csv("../data/Titanic/train.csv")
# 看头三行数据，默认5
# print(sklearn.head(3))
# 统计每一列数据特征 count(总计), mean(均值), std(方差), min(最小值), 25%, 50%, 75%, max
# print(sklearn.describe())

# 缺失值的填充, 只能针对数字类型的进行填充，像年龄这样的，用平均值
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
# print(sklearn['Age'])
# 打印Sex列去重后的值
# print(sklearn['Sex'].unique())

# 定位到Sex列，值 == male的变为0，把字符串转换为机器能理解的数据
titanic.loc[titanic['Sex'] == "male", "Sex"] = 0
titanic.loc[titanic['Sex'] == "female", "Sex"] = 1
# print(sklearn['Sex'].unique())

# 使用众数的来填充缺失值,因为众数可能有很多，所以取第一个
# print(sklearn['Embarked'].mode())
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
# 替换数值
titanic.loc[titanic['Embarked'] == "S", "Embarked"] = 0
titanic.loc[titanic['Embarked'] == "C", "Embarked"] = 1
titanic.loc[titanic['Embarked'] == "Q", "Embarked"] = 2
# 一样的数据补齐操作
titanic['Parch'] = titanic['Parch'].fillna(titanic['Parch'].mode()[0])

# 线性回归模型库
from sklearn.linear_model import LinearRegression
# 用于训练模型衡量，交叉验证：
# 例如：
# 1、把数据集分为"训练集"和"验证集"，
# 2、把"训练数据集"分成3份，代号：A,B,C。用A,B训练，用C验证；用A,C训练，用B验证；用B,C训练，用A验证。平均得到训练结果
from sklearn.model_selection import KFold

# 指定哪些特征用于机器学习
# 船舱等级、性别、年龄、随行兄弟姐妹，随行父母，船票价格，登船地点
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# 导入线性回归的模型
alg = LinearRegression()
# K折交叉验证 K-Folder Cross Validation
# n_splits 表示划分为几块（至少是2）
# shuffle 表示是否打乱划分，默认False，即不打乱
# random_state 表示是否固定随机起点，Used when shuffle == True.
kf = KFold(n_splits=3)

predictions = []
for train, test in kf.split(titanic):
    # 取出训练数据
    train_predictors = (titanic[predictors].iloc[train, :])
    # 数据的真实label
    train_target = titanic['Survived'].iloc[train]
    # 把线性回归应用于数据上，训练模型
    alg.fit(train_predictors, train_target)
    # 预测数据
    test_predictors = (titanic[predictors].iloc[test, :])
    # 对测试集进行预测
    test_predictions = alg.predict(test_predictors)
    # 把结果数据放入到数组中
    predictions.append(test_predictions)

# 由于线性回归得出来的值是基于[0，1]区间的任意一个值，所以需要处理成0和1
import numpy as np
# 训练三组数据，是三种结果，拼接所有结果
predictions = np.concatenate(predictions, axis=0)

predictions[predictions > .5] = "1"
predictions[predictions <= .5] = "0"
# 计算利用模型算出来的生存正确率
# 验证正确的数据
correct_test = predictions[predictions == titanic["Survived"]]
accuracy = len(correct_test) / len(predictions)
print("准确率：" + str(round(accuracy * 100, 4)) + "%")
