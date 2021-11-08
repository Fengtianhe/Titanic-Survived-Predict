# https://zhuanlan.zhihu.com/p/352062289

import tensorflow as tf
from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from tensorflow.saved_model import tag_constants

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
        data_train.columns = cols
        # print(data_train.head(10))
        label_train = data_train[['consumption_intention']]

        return data_train, label_train

    def data_processing(self, data_):
        # 训练集测试集都进行相同的处理
        data = data_[trainCols]
        return data

    def data(self):
        # 读数据
        train_data, label = self.get_data()
        # 处理数据
        # 训练集输入数据
        train = np.array(data_processing.data_processing(train_data))
        # 训练集标签
        train_label = np.array(label)

        return train, train_label


# 数据集
data_processing = DataProcessing()
x_train, y_train = data_processing.data()

learning_rate = 0.0005
training_epochs = 25
batch_size = 1000
display_step = 10
n_samples = x_train.shape[0]  # 数据量，行数
n_feature = x_train.shape[1]  # 特征数，列数
n_class = 1  # 二分类

# 设定占位符，模型变量和模型
x = tf.placeholder(tf.float32, shape=(None, n_feature), name='input')  # n_feature = 8 ，是指用8个特征进行预测
y = tf.placeholder(tf.float32, shape=(None, n_class))  # n_class = 2，目标是进行二分类
W = tf.Variable(tf.zeros([n_feature, n_class]), name='weight')
b = tf.Variable(tf.zeros([n_class]), name='bias')

pred = tf.matmul(x, W) + b

# 计算准确率和损失函数
correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
# reduce_mean是针对张量求平均值的一种方式，而reduce_sum是针对的一种求和
accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
# 优化参数
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

init = tf.initialize_all_variables()

start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    # 进行50次训练批次的训练，每批次使用50个数据进行训练
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)  # 在每个训练批次里面  我们都会使用所有的数据进行训练，所以这里会有个除式计算
        for i in range(total_batch):
            _, c = sess.run(
                [optimizer, cost],
                feed_dict={
                    x: x_train[i * batch_size:(i + 1) * batch_size],
                    y: y_train[i * batch_size:(i + 1) * batch_size]
                }
            )
            avg_cost = c / total_batch  # 单批次下的损失值
        plt.plot(epoch + 1, avg_cost, 'co')  # 把损失值变化 用plt图标来展示出来

        if (epoch + 1) % display_step == 0:  # 达到一定批次之后进行准确的输出展示
            print('Epoch:', (epoch + 1), ',cost=', avg_cost)
        # print('the Weight: ',sess.run(W),'  ,the bias:',sess.run(b))

    #     X_test = sess.run(tf.convert_to_tensor(x_test))
    #     y_test = sess.run(tf.convert_to_tensor(y_test))
    #     print('Testing Accuracy:', sess.run(accurary, feed_dict={x: x_test, y: y_test}))  # 这个地方如果这样用，往feed_dict中传的是张量，这是tf的理解难点
    #     plt.xlabel('Epoch')
    #     plt.ylabel('cost')
    #     plt.show()

    # 保存模型
    # https://blog.csdn.net/phynikesi/article/details/113933102
    print("start exporter model...")
    # signature设置
    inputs = {
        'input': tf.saved_model.utils.build_tensor_info(x)
    }
    outputs = {
        'y': tf.saved_model.utils.build_tensor_info(pred)
    }
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name='func'  # 名字自定义
    )

    # 模型保存
    # 这个目录在生成模型之前不要有，删除掉
    model_path = './train_seesion_model'
    builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature}, clear_devices=True)
    builder.save()

    # saver.restore(sess, tf.train.latest_checkpoint(model_path))

print('程序耗时：==> %.2fs' % (time.time() - start_time))
