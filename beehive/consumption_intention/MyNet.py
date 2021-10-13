import torch
from torch import nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 5),  # 输入层与第一隐层结点数设置，全连接结构
            nn.ReLU(),  # 第一隐层激活函数采用sigmoid
            nn.Linear(5, 5),  # 第一隐层与第二隐层结点数设置，全连接结构
            nn.ReLU(),  # 第一隐层激活函数采用sigmoid
            nn.Linear(5, 1),  # 第二隐层与输出层层结点数设置，全连接结构
        )
        self.opt = torch.optim.Adam(params=self.parameters(), lr=0.001)
        self.mls = nn.MSELoss()

    def forward(self, inputs):
        # 前向传播
        return self.fc(inputs)