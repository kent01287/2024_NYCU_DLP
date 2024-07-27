# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import TensorDataset

#assert False, "Not implemented yet!"

#multilayer layer of 3X3 kernel
#Residual learning

# block for ResNet34(解決了網路deeper ,效果比較差的問題,學習F(x)+x)
# stride =1 代表原圖大小不變
# stride =2 代表原圖大小變成一半
class BasicBlock(nn.Module): 
    def __init__(self, in_ch, out_ch, stride=1):         
        super().__init__() 
        self.conv = nn.Sequential(   
        nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), stride=stride, padding=1), 
        nn.BatchNorm2d(out_ch), 
        nn.ReLU(inplace=True), 
        nn.Conv2d(out_ch, out_ch,kernel_size=(3,3), stride=1, padding=1), 
        nn.BatchNorm2d(out_ch), 
        ) 
        #down 代表需不需要降維
        if stride!=1 or in_ch != out_ch: #stride !=1 或 in_ch != out_ch 代表要降低維度
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.down=None
    def forward(self, x): 
        out = self.conv(x)  
        if self.down is not None:
            residual=self.down(x)
        else:
            residual=x 
            
        out = residual + out
        out = F.relu(out) 
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, number): # 需要幾個 BasicBlock
        super().__init__()
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, 2)) #變一半
        for _ in range(1, number):
            layers.append(BasicBlock(out_ch, out_ch, 1)) #
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        out = self.blocks(x)
        return out

    
####### block for Unet#################
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.conv(x)
        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_ch,out_ch, up_in_ch=None, up_out_ch=None):
        super().__init__()
        if up_in_ch==None:
            up_in_ch=in_ch
        if up_out_ch==None:
            up_out_ch=out_ch
        
        self.up = nn.ConvTranspose2d(up_in_ch, up_out_ch, kernel_size=2, stride=2)
    
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    # x1-upconv , x2-downconv 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)



class ResNet34_UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.encoder1=nn.Sequential(
            nn.Conv2d(in_ch,64,kernel_size=(7,7),stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1)
        )# H/4
        self.encoder2=EncoderBlock(64,64,3) # H/8
        self.encoder3=EncoderBlock(64,128,4) # H/16
        self.encoder4=EncoderBlock(128,256,6)# H/32
        self.encoder5=EncoderBlock(256,512,3)# H/64
        ############# mid ##################
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.decoder1 = DecoderBlock(1024,512)
        self.decoder2 = DecoderBlock(512,256)
        self.decoder3 = DecoderBlock(256,128)
        self.decoder4 = DecoderBlock(128,64)
        self.decoder5 = DecoderBlock(128,64,64,64)
        self.last = nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size=(4,4), stride=4), #變成4倍
            nn.Conv2d(64, out_ch, kernel_size=(3,3), padding=1, bias=False)
        )
    def forward(self, x):
        ########## encoder #################
        # print("Initial input shape:", x.shape)  # [16, 3, 256, 256]
        e1 = self.encoder1(x)  # [16, 64, 64, 64]
        # print("Encoder1 output shape:", e1.shape)

        e2 = self.encoder2(e1)  # [16, 64, 32, 32]
        # print("Encoder2 output shape:", e2.shape)

        e3 = self.encoder3(e2)  # [16, 128, 16, 16]
        # print("Encoder3 output shape:", e3.shape)

        e4 = self.encoder4(e3)  # [16, 256, 8, 8]
        # print("Encoder4 output shape:", e4.shape)

        e5 = self.encoder5(e4)  # [16, 512, 4, 4]
        # print("Encoder5 output shape:", e5.shape)
        ########### mid ##########################
        mid = self.bridge(e5)  # [16, 1024, 2, 2]
        # print("mid output shape:", mid.shape)
        ############ decoder ##########################
        d1 = self.decoder1(mid, e5)  # [16, 512, 4, 4] 
        # print(f"d1 shape: {d1.shape}")

        d2 = self.decoder2(d1, e4)  # [16, 256, 8, 8]
        # print(f"d2 shape: {d2.shape}")

        d3 = self.decoder3(d2, e3)  # [16, 128, 16, 16]
        # print(f"d3 shape: {d3.shape}")

        d4 = self.decoder4(d3, e2)  # [16, 64, 32, 32]
        # print(f"d4 shape: {d4.shape}")

        d5 = self.decoder5(d4, e1)  # [16, 64, 64, 64]
        # print(f"d5 shape: {d5.shape}")

        output = self.last(d5)  # [16, 1, 256, 256]
        # print(f"output shape before flatten: {output.shape}")

        output = output.flatten(1)  # [16, 65536]
        # print(f"output shape after flatten: {output.shape}")
                
        return output

    
        