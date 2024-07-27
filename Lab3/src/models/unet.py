# Implement your UNet model here
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset

#assert False, "Not implemented yet!"

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
    #做up conv  decoder將特徵圖放大
    #這種設計的主要目的是保留高分辨率的信息，
    #這些信息在downsample過程中可能會丟失。通過(skip connection)，U-Net 能夠在decode過程中利用來自encoder的細節信息，從而提高最終的分割結果。
class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        ############### Conv  encoder block################
        
        self.conv1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = ConvBlock(512, 1024)
        
        ############ up conv  decoder block###############################
        self.upconv1=nn.ConvTranspose2d(1024,512, kernel_size=(2,2),stride=2) # H,W expand 兩倍
        self.deconv1 = ConvBlock(1024,512)
        
        self.upconv2=nn.ConvTranspose2d(512,256, kernel_size=(2,2),stride=2) 
        self.deconv2 = ConvBlock(512,256)
        
        self.upconv3=nn.ConvTranspose2d(256,128, kernel_size=(2,2),stride=2) 
        self.deconv3= ConvBlock(256,128)
        
        self.upconv4=nn.ConvTranspose2d(128,64, kernel_size=(2,2),stride=2) 
        self.deconv4= ConvBlock(128,64)
        
        self.deconvfinal = nn.Conv2d(64,out_ch,kernel_size=(1,1))
        
    def forward(self, x):
        ############# Encoder part #####################
        # Initial input shape: [16, 3, 256, 256]
        c1 = self.conv1(x)  # [16, 64, 256, 256]
        
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)

        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        
        p4 = self.pool4(c4)
        
        c5 = self.conv5(p4)
        
        ########crop and paste#########
        
        up1 = self.upconv1(c5) #[16, 512, 32, 32]
    
        x1 = torch.cat((c4, up1), dim=1)#[16, 1024, 32, 32]
        
        x2 = self.deconv1(x1)#[16, 512, 32, 32]
        
        
        up2 = self.upconv2(x2)#[16, 256, 64, 64]
        
        x3 = torch.cat((c3, up2), dim=1)#[16, 512, 64, 64]
        
        x4 = self.deconv2(x3)#[16, 256, 64, 64]
        
        
        up3 = self.upconv3(x4)#[16, 128, 128, 128]
        
        x5 = torch.cat((c2, up3), dim=1)#[16, 256, 128, 128]
        
        x6 = self.deconv3(x5)#[16, 128, 128, 128]
        
        
        up4 = self.upconv4(x6)#[16, 64, 256, 256]
       
        x7 = torch.cat((c1, up4), dim=1)#[16, 128, 256, 256]
        
        x8 = self.deconv4(x7)#[16, 64, 256, 256]
        
        ############final conv############
        
        output = self.deconvfinal(x8)#[16, 1, 256, 256]
        
        output = output.flatten(start_dim=1)#[16, 65536]
        
        return output

             
     
