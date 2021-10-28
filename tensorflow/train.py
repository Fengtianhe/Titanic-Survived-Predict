import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import time

cols = ["uid", "consumption_intention", "consumption_intention_origin", "ordinary_info_browse_uv",
        "ordinary_info_receive_resume_count", "fangke_invite_count", "goutongdou_invite_count", "zp_demand_intensity",
        "order_total_amount", "goutongdou_balance", "all_funds_balance", "dt"]
# 训练模型列
trainCols = ["ordinary_info_browse_uv", "ordinary_info_receive_resume_count", "fangke_invite_count",
             "goutongdou_invite_count", "zp_demand_intensity", "order_total_amount", "goutongdou_balance",
             "all_funds_balance"]


class DataProcessing(object):
    def __init__(self):
        pass

    def get_data(self):
        data_train = pd.read_csv('./train.csv/000000_0', header=None)
        print("训练数据量: %d" % data_train.size)
        print("训练维度数量: %d" % len(cols))
        data_train.columns = cols
        # print(data_train.head(10))
        label_train = data_train[['consumption_intention']]
        data_test = pd.read_csv('./test.csv/000000_0', header=None)
        print("测试数据量: %d" % data_test.size)
        data_test.columns = cols
        # 读取指定列，作为验证预测成功和失败的标识
        label_test = data_test[['consumption_intention']]
        return data_train, label_train, data_test, label_test

    def data_processing(self, data_):
        # 训练集测试集都进行相同的处理
        data = data_[trainCols]
        return data

    def data(self):
        # 读数据
        train_data, label, test_data, gender = self.get_data()
        # 处理数据
        # 训练集输入数据
        train = np.array(data_processing.data_processing(train_data))
        # 训练集标签
        train_label = np.array(label)
        # 测试集
        test = np.array(data_processing.data_processing(test_data))
        # 测试集标签，也就是测试集中预测的实际值
        test_label = np.array(gender)

        #         train = torch.from_numpy(train).float()
        #         train_label = torch.tensor(train_label).float()
        #         test = torch.tensor(test).float()
        #         test_label = torch.tensor(test_label)

        return train, train_label, test, test_label


# 数据集
data_processing = DataProcessing()
x_train, y_train, x_test, y_test = data_processing.data()

##############################################
# 建立模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(units=40, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("x_train.shape = %s " % str(x_train.shape))
print("y_train.shape = %s " % str(y_train.shape))

print("x_test.shape = %s " % str(x_test.shape))
print("y_test.shape = %s " % str(y_test.shape))

start_time = time.time()
train_history = model.fit(
    x=x_train,
    y=y_train,
    #     validation_data=(x_test, y_test),
    batch_size=30,
    epochs=1,
    verbose=2
)
print("训练完成")
print('程序耗时：==> %.2fs' % (time.time() - start_time))
print("====================================================================================")
# # 预测0，1
# predict_test = model.predict_classes(x_test)
# print(predict_test)

# # 真正例（TP）:实际上是正例(1)的数据点被标记为正例(1)
# TP = 0
# # 假反例（FN）:实际上是正例(1)的数据点被标记为返例(0)
# FN = 0
# # 真反例（TN）：实际上是反例的数据点被标记为反例
# TN = 0
# # 假正例（FP）：实际上是反例的数据点被标记为正例
# FP = 0
# for i, j in zip(predict_test, y_test):
#     # TP    predict 和 label 同时为1
#     TP += 1 if (i[0] == 1 & j[0] == 1) else 0
#     # TN    predict 和 label 同时为0
#     TN += 1 if (i[0] == 0 & j[0] == 0) else 0
#     # FN    predict 0 label 1
#     FN += 1 if (i[0] == 0 & j[0] == 1) else 0
#     # FP    predict 1 label 0
#     FP += 1 if (i[0] == 1 & j[0] == 0) else 0


# # 查准率
# p = TP / (TP + FP)
# # 召回率
# r = TP / (TP + FN)
# F1 = 2 * r * p / (r + p)
# acc = (TP + TN) / (TP + TN + FP + FN)
# print("TP = %s, TN = %s, FN = %s, FP = %s" % (TP, TN, FN, FP))
# print("召回率：=====> %.6f" % (r * 100))
# print("查准率：=====> %.6f" % (p * 100))
# print("F1：=====> %.6f" % (F1 * 100))
# print("acc：=====> %.6f" % (acc * 100))

print("====================================================================================")

# 预测概率
predict_percent_test = model.predict(x_test)
print(predict_percent_test)

# 真正例（TP）:实际上是正例(1)的数据点被标记为正例(1)
TP1 = 0
# 假反例（FN）:实际上是正例(1)的数据点被标记为返例(0)
FN1 = 0
# 真反例（TN）：实际上是反例的数据点被标记为反例
TN1 = 0
# 假正例（FP）：实际上是反例的数据点被标记为正例
FP1 = 0
for i, j in zip(predict_percent_test, y_test):
    # TP    predict 和 label 同时为1
    TP1 += 1 if ((i[0] >= 0.75) & (j[0] == 1)) else 0
    # TN    predict 和 label 同时为0
    TN1 += 1 if ((i[0] < 0.75) & (j[0] == 0)) else 0
    # FN    predict 0 label 1
    FN1 += 1 if ((i[0] < 0.75) & (j[0] == 1)) else 0
    # FP    predict 1 label 0
    FP1 += 1 if ((i[0] >= 0.75) & (j[0] == 0)) else 0

# 查准率
p1 = TP1 / (TP1 + FP1)
# 召回率
r1 = TP1 / (TP1 + FN1)
F11 = 2 * r1 * p1 / (r1 + p1)
acc1 = (TP1 + TN1) / (TP1 + TN1 + FP1 + FN1)
print("TP = %s, TN = %s, FN = %s, FP = %s" % (TP1, TN1, FN1, FP1))
print("召回率：=====> %.6f" % (r1 * 100))
print("查准率：=====> %.6f" % (p1 * 100))
print("F1：=====> %.6f" % (F11 * 100))
print("acc：=====> %.6f" % (acc1 * 100))

print("预测完成")