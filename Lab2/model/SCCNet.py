# implement SCCNet model

import torch
import torch.nn as nn

# reference paper: https://ieeexplore.ieee.org/document/8716937

# C個channels T個時間的二維DATA
# kernel size是
# Nu 個filter ， size為 C *Nt
#first second conv layer
# third is pooling
# fourth is softmax

# 定義平方層，用於提取數據的功率
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(x, 2)

# 定義 SCCNet 類別
class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=22, C=22, Nc=20, Nt=1, dropoutRate=0.5):
        super(SCCNet, self).__init__()

        # 第一卷積層
        self.conv1 = nn.Conv2d(1, Nu, (C, Nt), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(Nu)
        
        # 第二卷積層
        self.conv2 = nn.Conv2d(1, Nc, (Nu, 12), padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(Nc)
        
        # 平方層
        self.square = SquareLayer()
        
        # dropout
        self.dropout = nn.Dropout(dropoutRate)
        
        # pooling (use average pooling)
        self.pool = nn.AvgPool2d((1, 62), stride=(1, 12)) 
        
        # fc layer (softmax)
        self.fc = nn.Linear((timeSample - 62 + 1) * Nc, numClasses)

    def forward(self, x):
        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.square(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.square(x)
        
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.fc(x)
        
        return x