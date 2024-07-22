# implement SCCNet model
import torch
import torch.nn as nn
# reference paper: https://ieeexplore.ieee.org/document/8716937



#(sample,電極,T)
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x**2

class LogLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.log(x)

# 定義 SCCNet 類別
class SCCNet(nn.Module):


    
    def __init__(self, numClasses=4, timeSample=438, Nu=22, C=22, Nc=20, Nt=16, dropoutRate=0.5):
        super(SCCNet, self).__init__()
        
        
        self.conv1 = nn.Conv2d(1, Nu, (C, Nt), padding=0) #(batch size ,kernel ,height ,width)
        self.bn1 = nn.BatchNorm2d(Nu)
        
        
        
        self.conv2 = nn.Conv2d(1, Nc, (Nu, 12), padding=(0,6))
        self.bn2 = nn.BatchNorm2d(Nc)
        
        # square activation function
        self.square = SquareLayer()
        # dropout
        self.dropout = nn.Dropout(dropoutRate)
        # pooling (use average pooling)
        
        
        
        self.pool = nn.AvgPool2d((1, 64), stride=(1, 12)) 
        self.logLayer =LogLayer()
        # output shape (20,T/12)
        afterPoolSize = (timeSample-64)//12 
        self.fc = nn.Linear(Nc*afterPoolSize, numClasses) #620
        
     
    def forward(self, x):
        #input shape[batch,22,438]
        #print("Input shape:", x.shape)
        x = x.unsqueeze(1)  # (batch , 1 ,22,438)
        #print("After unsqueeze:", x.shape)
       
        ##############first layer####################
        x = self.conv1(x) #  (batch,22,1,423) width -Nt+1
        #print("After conv1:", x.shape)
        x = self.bn1(x)
        #print("After bn1:", x.shape)
        x = self.dropout(x)
        x = x.permute(0,2, 1, 3) # [batch, 1, 22, 423] 
        
        ##############Second layer####################
        #print("After permute:", x.shape)
        x = self.conv2(x) #[batch, 20, 1, 424]
        #print("After conv2:", x.shape)
        x = self.bn2(x) #[batch, 20, 1, 424]
        #print("After bn2:", x.shape)
        x = self.square(x) # [batch, 20, 1, 424]
        #print("After square2:", x.shape)
        x = self.dropout(x) # [batch, 20, 1, 424]
        #print("After dropout:", x.shape)
        x = x.permute(0,2, 1, 3)
        
        ##############third layer####################
        x = self.pool(x) # [batch, 1, 20, 31]
        #print("After pool:", x.shape)
        x=self.logLayer(x) # [batch, 1, 20, 31]
        
        ##############forth layer##############
        x = x.flatten(1)   #[batcg, 620]
        #print("After flatten:", x.shape)
        x = self.fc(x) #[batch,4]
        #print("After fc:", x.shape)

        return x
    
