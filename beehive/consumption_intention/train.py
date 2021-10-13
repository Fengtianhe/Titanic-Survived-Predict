import logging

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

dt = time.strftime('%Y%m%d', time.localtime())
logging.basicConfig(filename='run_' + str(dt) + '.log', level=logging.INFO)
logger = logging.getLogger(__name__)
# 数据表头

cols = ["uid", "consumption_intention", "consumption_intention_origin", "ordinary_info_browse_uv",
        "ordinary_info_receive_resume_count", "fangke_invite_count", "goutongdou_invite_count", "zp_demand_intensity",
        "order_total_amount", "goutongdou_balance", "all_funds_balance", "dt"]
# 训练模型列
trainCols = ["ordinary_info_browse_uv", "ordinary_info_receive_resume_count", "fangke_invite_count",
             "goutongdou_invite_count", "zp_demand_intensity", "order_total_amount", "goutongdou_balance",
             "all_funds_balance"]

# 显示所有列
pd.set_option('display.max_columns', None)
# 表头不换行
pd.set_option('expand_frame_repr', False)


class DataProcessing(object):
    def __init__(self):
        pass

    def get_data(self):
        data_train = pd.read_csv('train.csv/000000_0', header=None)
        print("训练数据量: %d" % data_train.size)
        print("训练维度数量: %d" % len(cols))
        data_train.columns = cols
        # print(data_train.head(10))
        label = data_train[['consumption_intention']]
        data_test = pd.read_csv('test.csv/000000_0', header=None)
        print("测试数据量: %d" % data_test.size)
        data_test.columns = cols
        # 读取指定列，作为验证预测成功和失败的标识
        gender = pd.read_csv('test.csv/000000_0', usecols=[1])
        return data_train, label, data_test, gender

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

        train = torch.from_numpy(train).float()
        train_label = torch.tensor(train_label).float()
        test = torch.tensor(test).float()
        test_label = torch.tensor(test_label)

        return train, train_label, test, test_label


# RuntimeError: mat1 and mat2 shapes cannot be multiplied (2598951x8 and 11x7)
# 这个报错就是模型的维度错，nn.Linear(input, output) 是全连接层 ,
# input 应该是特征数，就是你输入的维度，
# 比如你每个样本的特征数量是100，你input 就是100，
# output 就是你隐藏层的神经元个数，
# 就是你想把100个特征压缩成64个，那output 就是 64 。
# 你先确认一下 你输入的数据的维度吧，比如你有5000个样本，每个样本的长度是100，input 就是100 ，看你的报错，你的特征数是8 ？ 那input 就是8。
# https://www.cnblogs.com/wangqinze/p/13424368.html
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(11, 7), # 网上泰坦尼克的方式，用11个维度去预测的
            nn.Linear(8, 5),  # 输入层与第一隐层结点数设置，全连接结构
            nn.ReLU(),  # 第一隐层激活函数采用sigmoid
            nn.Linear(5, 5),  # 第一隐层与第二隐层结点数设置，全连接结构
            nn.ReLU(),  # 第一隐层激活函数采用sigmoid
            nn.Linear(5, 1),  # 第二隐层与输出层层结点数设置，全连接结构
        )
        # Adam 优化算法是深度学习问题中一种非常流行的选择算法。这是随机梯度下降算法的扩展，但与 SGD 算法不同，Adam 优化器在训练期间并不保持相同的学习速率。
        self.opt = torch.optim.Adam(params=self.parameters(), lr=0.001)
        self.mls = nn.MSELoss()

    def forward(self, inputs):
        # 前向传播
        return self.fc(inputs)

    def train(self, inputs, y):
        # 训练
        self.opt.zero_grad()  # 清除梯度
        out = self.forward(inputs) # 训练模型
        loss = self.mls(out, y) # 计算损失
        loss.backward()  # 计算梯度，误差回传
        self.opt.step()  # 根据计算的梯度，更新网络中的参数
        print("损失值 => " + str(loss.data.numpy()))

    def test(self, x, y):
        # 测试
        # 将variable张量转为numpy
        # out = self.fc(x).data.numpy()
        count = 0
        out = self.fc(x)
        sum = len(y)
        for i, j in zip(out, y):
            i = i.detach().numpy()
            j = j.detach().numpy()
            loss = abs((i - j)[0])
            if loss < 0.3:
                count += 1
        # 误差0.3内的正确率
        print("正确率：=====> %.6f" % (count * 100 / sum))


if __name__ == '__main__':
    # 耗时
    start_time = time.time()
    data_processing = DataProcessing()
    train_data, train_label, test_data, test_label = data_processing.data()

    net = MyNet()
    count = 0
    # range中的参数可以调整，多训练几次，用mac电脑训练2次花了30s
    for i in range(2):
        # 为了减小电脑压力,分批训练 20000个训练一次  ## 正确的做法应该是用batch
        for n in range(len(train_data) // 50000 + 1):
            batch_data = train_data[n * 100: n * 100 + 100]
            batch_label = train_label[n * 100: n * 100 + 100]
            net.train(train_data, train_label)

    # 测试模型
    print("测试模型")
    net.test(test_data, test_label)

    print("导出模型")
    torch.save(net.state_dict(), 'model/model.pth')

    print('程序耗时：==> %.2fs' % (time.time() - start_time))
