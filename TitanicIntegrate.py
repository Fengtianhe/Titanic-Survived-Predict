# -*- coding: utf-8 -*-
# 利用随机森林验证泰坦尼克生存数据
# %%
import numpy as np
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

# 随机森林算法
# 随机森林不止对数据随机，也对特征随机，比如提供7个特征，那指定算法会取5个随机特征进行训练
# 生成多个决策树的众数
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

feature = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

algorithms = [
    [RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2), feature],
    [LogisticRegression(random_state=1, max_iter=1000), feature]
]

kf = KFold(n_splits=3)

predictions = []
for train, test in kf.split(titanic):
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train, :], train_target)
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    # 可以使用平均值，可以使用权重
    test_predictions = (full_test_predictions[0] * 3 + full_test_predictions[1]) / 4
    # test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

accuracy = len(predictions[predictions == titanic['Survived']]) / len(predictions)
print("准确率：" + str(round(accuracy * 100, 4)) + "%")
