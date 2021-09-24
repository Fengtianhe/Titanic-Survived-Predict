import torch
from torch import nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(11, 7), # 网上泰坦尼克的方式，用11个维度去预测的
            nn.Linear(8, 5),
            nn.Sigmoid(),
            nn.Linear(5, 5),
            nn.Sigmoid(),
            nn.Linear(5, 1),
        )
        self.opt = torch.optim.Adam(params=self.parameters(), lr=0.001)
        self.mls = nn.MSELoss()

    def forward(self, inputs):
        # 前向传播
        return self.fc(inputs)