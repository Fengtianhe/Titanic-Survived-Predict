# 预处理函数
# 参数：
# x 用户jar包中，数据处理类的predictOnlineBefore函数封装的数据，类型包括str，bytes，numpy.array，kwargs
# kwargs 用户jar包中，数据处理类的predictOnlineBefore函数封装的参数
# 返回值：
# 模型执行的输入数据
import torch
from MyNet import MyNet
import logging

logger = logging.getLogger(__name__)


def preprocess(x, **kwargs):
    logger.info("====================================================================")
    logger.info("进入到预处理函数，接收到数据" + str(x) + "参数 " + str(kwargs))
    x = str(x)[1:-1].split(" ")
    x = [float(i) for i in x]
    logging.info("处理后的数据 ==========》" + str(x))
    x = torch.tensor(x, dtype=torch.float32)
    return x


# 后处理函数
# 参数：
# x 模型执行后的输出数据，即model(data)所得得结果
# kwargs 用户jar包中，数据处理类的predictOnlineBefore函数封装的参数
# 返回值：
# 用户jar包中，数据处理类的predictOnlineAfter函数的输入数据类型
def postprocess(x, **kwargs):
    return x


# 模型加载函数，用户自定义
# 参数：
# 返回值：
# 加载好的模型，用于模型推理
def load_model():
    model = MyNet()
    model.load_state_dict(torch.load('./model.pth'))
    return model


# 自定义推理执行函数
# 参数：
# model 模型对象
# x 预处理后的数据，即preprocess函数所得得结果
# kwargs 用户jar包中，数据处理类的predictOnlineBefore函数封装的参数
# 返回值：
# 模型推理处理结果
def run_model(model, x, **kwargs):
    logger.info("===========run_model============")

    return model(x)
