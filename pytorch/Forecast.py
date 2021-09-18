# -*- coding: utf-8 -*-
import torch
from torch import nn


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(titanic_train_data_X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    model_dict = model.load_state_dict(torch.load('./model.pth'))
