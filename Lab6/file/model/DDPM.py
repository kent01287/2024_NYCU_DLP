from diffusers import DDPMScheduler,UNet2DModel
import torch
import torch.nn as nn
'''
    implement 1 : Noise Schedule 每個時間點增加了多少噪音  [ linear cosine]
    implement 2 : Time Embeddings 使用位置編碼或可學習的嵌入將時間步長轉換為特徵向量。
    class embedded 是linear layer 與圖像數據 time step一起丟入訓練
    "DownBlock2D",  # a regular ResNet downsampling block
    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    "UpBlock2D",  # a regular ResNet upsampling block
'''
class DDPM(nn.Module):
    def __init__(self,num_class=24, class_emb_size=512):  
        super(DDPM, self).__init__()
        self.unet_model= UNet2DModel(
        sample_size=64,  
        in_channels=3,   
        out_channels=3,  
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512,512),
        #norm_num_groups=32,
        down_block_types=("DownBlock2D", "DownBlock2D","DownBlock2D" ,"DownBlock2D","DownBlock2D","AttnDownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D","AttnUpBlock2D"),
    )
        self.unet_model.class_embedding = nn.Linear(num_class, class_emb_size)
    
    def forward(self, x, t,label):
        output = self.unet_model(x, t,label.float()).sample
        return output
    
    