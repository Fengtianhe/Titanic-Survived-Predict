import csv
import tensorflow as tf
import numpy as np
import random
import sys
import pandas as pd
from pandas import DataFrame

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

trainFilePath = './data/Titanic/train.csv'

trainSize = 0

global trainData, targetData, classifier
with open(trainFilePath, 'rb') as trainFile:
    csvReader = csv.reader(trainFile)
    dataList = [data for data in csvReader]
    dataSize = len(dataList) - 1
    trainData = np.ndarray((dataSize, 4), dtype=np.float32)
    targetData = np.ndarray((dataSize, 1), dtype=np.int32)
    trainDataFrame = DataFrame(dataList[1:], columns=dataList[0])
    trainDataFrame_fliter = trainDataFrame.loc[:, ['Pclass', 'Sex', 'SibSp', 'Fare', 'Survived']]
    for i in range(dataSize):
        thisData = np.array(trainDataFrame_fliter.iloc[i])
        Pclass, Sex, SibSp, Fare, Survived = thisData
        Pclass = float(Pclass)
        Sex = 0 if Sex == 'female' else 1
        SibSp = float(SibSp)
        Fare = float(Fare)
        Survived = int(Survived)
        print(Pclass, Sex, SibSp, Fare, Survived)
        trainData[i, :] = [Pclass, Sex, SibSp, Fare]
        targetData[i, :] = [Survived]
        print(thisData)
    print(trainData)
    print(targetData)
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=2)


# 我们将训练数据和标签包装成一个二元组，并返回
def get_train_inputs():
    x = tf.constant(trainData)
    y = tf.constant(targetData)
    print(x)
    print(y)
    return x, y


# 训练模型
classifier.fit(input_fn=get_train_inputs, steps=2000)

# 检查准确度
accuracy_score = classifier.evaluate(input_fn=get_train_inputs, steps=1)["accuracy"]
print("accuracy:", accuracy_score)
