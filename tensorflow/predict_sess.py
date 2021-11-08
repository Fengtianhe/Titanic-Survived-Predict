import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.saved_model import tag_constants
from tensorflow.contrib.framework.python.framework import checkpoint_utils

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
        data_test = pd.read_csv('./test.csv/000000_0', header=None)
        print("测试数据量: %d" % data_test.size)
        data_test.columns = cols
        # 读取指定列，作为验证预测成功和失败的标识
        label_test = data_test[['consumption_intention']]
        return data_test, label_test

    def data_processing(self, data_):
        # 训练集测试集都进行相同的处理
        data = data_[trainCols]
        return data

    def data(self):
        # 读数据
        test_data, gender = self.get_data()
        # 测试集
        test = np.array(data_processing.data_processing(test_data))
        # 测试集标签，也就是测试集中预测的实际值
        test_label = np.array(gender)

        return test, test_label


# 数据集
data_processing = DataProcessing()
x_test, y_test = data_processing.data()

with tf.Session() as sess:
    var_list = checkpoint_utils.list_variables('./train_seesion_model')
    for v in var_list:
        print(v)

    # 不用执行初始化
    meta_graph_def = tf.saved_model.load(sess, [tag_constants.SERVING], './train_seesion_model')

    graph = tf.get_default_graph()

#     x = sess.graph.get_tensor_by_name('input:0')
#     y = sess.graph.get_tensor_by_name('y:0')

#     scores = sess.run(y, feed_dict={x: x_test})
#     print(scores)
