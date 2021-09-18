import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

"""
一，准备数据
"""

# 数据读取
train_data = pd.read_csv('../data/Titanic/train.csv')
test_data = pd.read_csv('../data/Titanic/test.csv')
test_datay = pd.read_csv('../data/Titanic/gender_submission.csv')
# print(train_data.head(10))  #打印训练数据前十个

train_data.info()  # 查看训练数据有没有未知的的
test_data.info()  # 查看测试数据有没有未知的的

# #查看各部分分布情况
#
# #幸存情况
# ax = train_data['Survived'].value_counts().plot(kind = 'bar',figsize = (12,8),fontsize =15,rot = 0)
# #value_counts是查询有多少个不同值且每个不同值有多少个重复的
# ax.set_ylabel('Counts',fontsize = 15)
# ax.set_xlabel('Survived',fontsize = 15)
# plt.show()
#
# #年龄分布情况
# ax = train_data['Age'].plot(kind = 'hist',bins = 20,color = 'purple',figsize = (12,8),fontsize = 15)
# """
# hist方法常用的参数有以下几个
# 1. bins,控制直方图中的区间个数
# 2. color,指定柱子的填充色
# 3. edgecolor, 指定柱子边框的颜色
# 4. density,指定柱子高度对应的信息，有数值和频率两种选择
# 5. orientation，指定柱子的方向，有水平和垂直两个方向
# 6. histtype，绘图的类型
# """
# ax.set_ylabel('Frequency',fontsize = 15)
# ax.set_xlabel('Age',fontsize = 15)
# plt.show()
#
# #年龄和label的相关性
# ax = train_data.query('Survived == 0')['Age'].plot(kind = 'density',figsize = (12,8),fontsize = 15)
# #使用python.query()函数对数据框进行（挑选行）的操作
# train_data.query('Survived == 1')['Age'].plot(kind = 'density',figsize = (12,8),fontsize = 15)
# ax.legend(['Survived ==0','Survived ==1'],fontsize = 12)
# #plt.legend（）函数主要的作用就是给图加上图例，plt.legend([x,y,z])里面的参数使用的是list的的形式将图表的的名称喂给这和函数。
# ax.set_ylabel('Density',fontsize = 15)
# ax.set_xlabel('Age',fontsize = 15)
# plt.show()
#
"""
数据预处理
"""


def preprocessing(dfdata):
    dfresult = pd.DataFrame()  # 存储结果
    # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。

    # Pclass处理
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    # 对Pclass进行get_dummies,将该特征离散化
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)
    # concat函数是在pandas底下的方法，可以将数据根据不同的轴作简单的融合,axis： 需要合并链接的轴，0是行，1是列

    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')
    # pandas.isna(obj)检测array-like对象的缺失值

    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    # DataFrame.columns属性以返回给定 DataFrame 的列标签。

    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return dfresult


# 获得训练x,y
x_train = preprocessing(train_data).values
y_train = train_data[['Survived']].values

# 获得测试x,y
x_test = preprocessing(test_data).values
y_test = test_datay[['Survived']].values

# print("x_train.shape =", x_train.shape )
# print("x_test.shape =", x_test.shape )
# print("y_train.shape =", y_train.shape )
# print("y_test.shape =", y_test.shape )
#
"""
进一步使用DataLoader和TensorDataset封装成可以迭代的数据管道。
"""
dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float()),
                      shuffle=True, batch_size=8)
dl_valid = DataLoader(TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float()),
                      shuffle=False, batch_size=8)
#
# # #测试数据管道
# for features,labels in dl_valid:
#     print(features,labels)
#     break
#
"""
二，定义模型
"""


def creat_net():
    net = nn.Sequential()
    net.add_module("linear1", nn.Linear(15, 20))
    net.add_module("relu1", nn.ReLU())
    net.add_module("linear2", nn.Linear(20, 15))
    net.add_module("relu2", nn.ReLU())
    net.add_module("linear3", nn.Linear(15, 1))
    net.add_module("sigmoid", nn.Sigmoid())
    return net


net = creat_net()
# print(net)

"""
三，训练模型
"""

from sklearn.metrics import accuracy_score

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
metric_func = lambda y_pred, y_true: accuracy_score(y_true.data.numpy(), y_pred.data.numpy() > 0.5)
# lambda表达式是起到一个函数速写的作用。允许在代码内嵌入一个函数的定义。
# accuracy_score是分类准确率分数是指所有分类正确的百分比。
metric_name = "accuracy"
# metric就是准确率


epochs = 10
log_step_freq = 30
dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])

for epoch in range(1, epochs + 1):
    # 开始训练
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1
    for step, (features, labels) in enumerate(dl_train, 1):
        optimizer.zero_grad()

        # 正向传播
        predictions = net(features)
        loss = loss_func(predictions, labels)
        metric = metric_func(predictions, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 打印batch日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step % log_step_freq == 0:
            print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") % (step, loss_sum / step, metric_sum / step))

    # 验证循环
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1
    for val_step, (features, labels) in enumerate(dl_valid, 1):
        predictions = net(features)
        val_loss = loss_func(predictions, labels)
        val_metric = metric_func(predictions, labels)

        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()

    # 记录日志
    info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step)
    dfhistory.loc[epoch - 1] = info

    # 打印epoch级别日志
    print((
                      "\nEPOCH = %d, loss = %.3f," + metric_name + "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f") % info)

"""
四，评估模型
"""


def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


plot_metric(dfhistory, "loss")
plot_metric(dfhistory, "accuracy")

"""
五，使用模型
"""
y_pred_probs = net(torch.tensor(x_test[0:10]).float()).data
y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))

# """
# # 六，保存模型
# """
# #保存模型参数(推荐)
# print(net.state_dict().keys())
# # 保存模型参数
# torch.save(net.state_dict(), "./data/net_parameter.pkl")
# net_clone = creat_net()
# net_clone.load_state_dict(torch.load("./data/net_parameter.pkl"))
# net_clone.forward(torch.tensor(x_test[0:10]).float()).data
#
# #保存完整模型(不推荐)
# torch.save(net, './data/net_model.pkl')
# net_loaded = torch.load('./data/net_model.pkl')
# net_loaded(torch.tensor(x_test[0:10]).float()).data
