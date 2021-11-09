import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import time
import os

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
        label_test =  data_test[['consumption_intention']]
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
model.add(Dense(units=1,kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("x_train.shape = %s " % str(x_train.shape))
print("y_train.shape = %s " % str(y_train.shape))

print("x_test.shape = %s " % str(x_test.shape))
print("y_test.shape = %s " % str(y_test.shape))

start_time = time.time()

# 不用定义文件名，默认保存为saved_model.pd文件
save_path = "/tmp/zhaopin/beehive/consumption_intention"
train_history = model.fit(
    x=x_train,
    y=y_train,
#     validation_data=(x_test, y_test),
    batch_size=30,
    epochs=1,
    verbose=2
)
model.save(save_path)

print("训练完成")
print('程序耗时：==> %.2fs' % (time.time() - start_time))
print("====================================================================================")

# # 预测概率
# predict_percent_test = model.predict(x_test)

# # 这里是模型相关指标，真正训练是不需要打印
# print("|阈值|真正例TP|真反例TN|假正例FP|假反例FN|准确率|召回率|查准率|F1 Score|")
# print("|---|-------|-------|-------|-------|-----|-----|-----|--------|")
# for threshold in range(0, 100, 5):
#     # 真正例（TP）:实际上是正例(1)的数据点被标记为正例(1)
#     TP1 = 0
#     # 假反例（FN）:实际上是正例(1)的数据点被标记为返例(0)
#     FN1 = 0
#     # 真反例（TN）：实际上是反例的数据点被标记为反例
#     TN1 = 0
#     # 假正例（FP）：实际上是反例的数据点被标记为正例
#     FP1 = 0
# #     print(predict_percent_test)
# #     print(y_test)
#     for i, j in zip(predict_percent_test, y_test):
#         # TP    predict 和 label 同时为1
#         TP1 += 1 if ((i[0] >= (threshold / 100)) & (j[0] == 1)) else 0
#         # TN    predict 和 label 同时为0
#         TN1 += 1 if ((i[0] < (threshold / 100)) & (j[0] == 0)) else 0
#         # FN    predict 0 label 1
#         FN1 += 1 if ((i[0] < (threshold / 100)) & (j[0] == 1)) else 0
#         # FP    predict 1 label 0
#         FP1 += 1 if ((i[0] >= (threshold / 100)) & (j[0] == 0)) else 0


# #     print("%s--%s--%s--%s--%s" % (threshold, TP1, TN1, FP1, FN1))
#     # 查准率
#     p1 = TP1 / (TP1 + FP1)
#     # 召回率
#     r1 = TP1 / (TP1 + FN1)
#     F11 = 2 * r1 * p1 / (r1 + p1)
#     acc1 = (TP1 + TN1) / (TP1 + TN1 + FP1 + FN1)
#     print("|%s|%s|%s|%s|%s|%.6f|%.6f|%.6f|%.6f|" % ((threshold / 100), TP1, TN1, FP1, FN1, acc1, r1, p1, F11))
# print("|---|-------|-------|-------|-------|-----|-----|-----|--------|")
# print("预测完成")