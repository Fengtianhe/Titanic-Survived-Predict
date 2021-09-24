import time

import numpy as np
import pandas as pd
import torch
from beehive.MyNet import MyNet
import logging

logger = logging.getLogger(__name__)
if __name__ == '__main__':
    # 耗时
    start_time = time.time()

    model = MyNet()
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    # 读取数据并转换成Tensor类型
    data_predict = pd.read_csv('./predict.csv', header=None)
    print(type(data_predict))
    my_array = np.array(data_predict)
    data = "[24. 0. 0. 0. 7. 0. 0. 0.]"
    x = data[1:-1].split(" ")
    x = [float(i) for i in x]
    print(x)
    my_tensor = torch.tensor(x, dtype=torch.float32)
    logging.info("处理后的数据")
    print(my_tensor)
    output = model(my_tensor)
    print(output)

    print('程序耗时：==> %.2fs' % (time.time() - start_time))
