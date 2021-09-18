# -*- coding: utf-8 -*-
# 利用随机森林验证泰坦尼克生存数据
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

# 随机森林算法
# 随机森林不止对数据随机，也对特征随机，比如提供7个特征，那指定算法会取5个随机特征进行训练
# 生成多个决策树的众数
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# 一个森林有多个决策树决定，每个决策树都需要分裂，那分裂到什么时候停止就使用 min_samples_leaf 指定叶子叶子节点的个数，min_samples_split 指定样本的个数
# n_estimators 决策树数量决定准确率
# 放松决策树分裂的参数配置
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
# 进行三次交叉验证
kf = KFold(n_splits=3)
scores = cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf.split(titanic))

print("准确率：" + str(round(scores.mean() * 100, 4)) + "%")
