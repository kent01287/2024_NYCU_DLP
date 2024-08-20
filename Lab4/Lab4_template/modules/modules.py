import torch.nn as nn
import torch
from .layers import DepthConvBlock, ResidualBlock
from torch.autograd import Variable

#from module import * 只有列表的會被import
__all__ = [
    "Generator",
    "RGB_Encoder",
    "Gaussian_Predictor",
    "Decoder_Fusion",
    "Label_Encoder"
] 
'''
'''

class Generator(nn.Sequential):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__(
            DepthConvBlock(input_nc, input_nc),
            ResidualBlock(input_nc, input_nc//2),
            DepthConvBlock(input_nc//2, input_nc//2),
            ResidualBlock(input_nc//2, input_nc//4),
            DepthConvBlock(input_nc//4, input_nc//4),
            ResidualBlock(input_nc//4, input_nc//8),
            DepthConvBlock(input_nc//8, input_nc//8),
            nn.Conv2d(input_nc//8, 3, 1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
#通過多層殘差塊和深度卷積塊逐步將 RGB 圖像轉換為高維特徵表示。
#每個塊的作用是提取和增強特徵，最終的卷積層將特徵通道數轉換為所需的輸出特徵數，    
class RGB_Encoder(nn.Sequential):
    def __init__(self, in_chans, out_chans): #(3,128) transfer RGB to feature domain
        super(RGB_Encoder, self).__init__(
            ResidualBlock(in_chans, out_chans//8),
            DepthConvBlock(out_chans//8, out_chans//8),
            ResidualBlock(out_chans//8, out_chans//4),
            DepthConvBlock(out_chans//4, out_chans//4),
            ResidualBlock(out_chans//4, out_chans//2),
            DepthConvBlock(out_chans//2, out_chans//2),
            nn.Conv2d(out_chans//2, out_chans, 3, padding=1),
        )  
        
    def forward(self, image):
        return super().forward(image)
    

    
    
    
class Label_Encoder(nn.Sequential):
    def __init__(self, in_chans, out_chans, norm_layer=nn.BatchNorm2d):
        super(Label_Encoder, self).__init__(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_chans, out_chans//2, kernel_size=7, padding=0),
            norm_layer(out_chans//2),
            nn.LeakyReLU(True),
            ResidualBlock(in_ch=out_chans//2, out_ch=out_chans)
        )  
        
    def forward(self, image):
        return super().forward(image)
    
#將 q(z|x) 將x sample 出來 透過神經網路得到 mu , log var 
class Gaussian_Predictor(nn.Sequential):
    def __init__(self, in_chans=48, out_chans=96):
        super(Gaussian_Predictor, self).__init__(
            ResidualBlock(in_chans, out_chans//4),
            DepthConvBlock(out_chans//4, out_chans//4),
            ResidualBlock(out_chans//4, out_chans//2),
            DepthConvBlock(out_chans//2, out_chans//2),
            ResidualBlock(out_chans//2, out_chans),
            nn.LeakyReLU(True),
            nn.Conv2d(out_chans, out_chans*2, kernel_size=1)
        )
    #VAE參數化是分布是不可微分的   
    #重參數化的技巧是假設 prior distribition 是 Normal distrbution
    #，因此可以將 latent space 以 z=mu+std*eplision表示
    #表示，則使得分布與網路參數分布無關，使其可以計算 Gradient。
    def reparameterize(self, mu, logvar): #直接從後驗分佈採z比較困難 所以使用reparameterize 
        #透過encoder 得到 mu 和 log var 使用 reparameterize 得到隨機變量z (要是normal distribution)
        # TODO
        # 生成大小mu 的 normal distribution
        epsilon = torch.randn_like(mu)
        
        # 算標準差
        std = torch.exp(0.5 * logvar)
        z = mu + std * epsilon
        return z      
        #raise NotImplementedError

    def forward(self, img, label): #img 靜止圖片 #label 影片的那一偵 
        feature = torch.cat([img, label], dim=1)
        parm = super().forward(feature)
        mu, logvar = torch.chunk(parm, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar
    
    
class Decoder_Fusion(nn.Sequential):
    def __init__(self, in_chans=48, out_chans=96):
        super().__init__(
            DepthConvBlock(in_chans, in_chans),
            ResidualBlock(in_chans, in_chans//4),
            DepthConvBlock(in_chans//4, in_chans//2),
            ResidualBlock(in_chans//2, in_chans//2),
            DepthConvBlock(in_chans//2, out_chans//2),
            nn.Conv2d(out_chans//2, out_chans, 1, 1)
        )
    #img來自 RGB Encoder 主要是 靜態圖像的特徵    
    #label來自 label Encoder 主要是 骨架動作的特徵  
    #parm 來自 Gaussian_Predictor的mu and logVar 代表潛在空間
    def forward(self, img, label, parm):
        feature = torch.cat([img, label, parm], dim=1)
        return super().forward(feature)
    

    
        
    
if __name__ == '__main__':
    pass
