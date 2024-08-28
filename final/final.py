import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import requests
from io import BytesIO
from PIL import Image

import torchvision.transforms as transforms

import random
######### add noise ###############
def add_noise(x,noise_level,noise_type):

    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level/255, x.shape) #mu=0

        noisy = torch.clamp(noisy,0,1) #限制0~1


    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x)/noise_level

    return noisy
######## modify network ###################

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, use_mask=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.use_mask = use_mask
        self.mask = self.make_mask(kernel_size)

    def make_mask(self, kernel_size):
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2, kernel_size // 2] = 0
        return mask.view(1, 1, kernel_size, kernel_size)

    def forward(self, x):
        if self.use_mask:
            self.weight.data *= self.mask.to(self.weight.device)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)

class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=48, mask_prob=0.5):
        super(Network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mask_prob = mask_prob

        # 使用普通的卷積層
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        # 使用掩碼卷積層
        self.conv2 = MaskedConv2d(chan_embed, chan_embed, 3, padding=1, use_mask=True)
        # 最後一層仍然使用普通卷積
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))  # [1, 48, 128, 128]

        
        if self.training:  
            self.conv2.use_mask = random.random() < self.mask_prob
        else:
            self.conv2.use_mask = False  

        x = self.act(self.conv2(x))  
        x = self.conv3(x)        
        return x
    ####### down sampling process
def pair_downsampler(img):
    #img has shape B C H W
    c = img.shape[1] #channal數目

    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    #print(output1.shape)
    #print(output2.shape)
    return output1, output2
def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor: #應該返回Tensor
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

def loss_func(model,noisy_img):
  ######### 先downSampler 再 denoise####################
    noisy1, noisy2 = pair_downsampler(noisy_img)

    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)

    loss_res = 1/2*(mse(noisy1,pred2)+mse(noisy2,pred1)) #圖1 與去躁後的圖二 + 圖二與去躁後的圖一



    #只使用res 會overfitting
###############先 denoise 再 downsampler #######################
    #所以提出一致性損失
    noisy_denoised =  noisy_img - model(noisy_img) #先denoise
    denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons=1/2*(mse(pred1,denoised1) + mse(pred2,denoised2))

    loss = loss_res + loss_cons

    ############## new #####################
    '''
    #return loss

    noisy1=random_subsample(noisy_img)
    noisy2=random_subsample(noisy_img)

    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)

    loss_new = 1/2*(mse(noisy1,pred2)+mse(noisy2,pred1)) #圖1 與去躁後的圖二 + 圖二與去躁後的圖一
    #loss = loss_res + loss_cons + loss_new
    '''
    ########################## try quad     ######################
    # 示例使用
    '''
    noisy1, noisy2, noisy3, noisy4 = quad_downsampler(noisy_img)

    # 使用模型進行去噪預測
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    pred3 = noisy3 - model(noisy3)
    pred4 = noisy4 - model(noisy4)

    # 計算損失，使用每個區塊與其他區塊去噪後結果的損失
    loss_quad = (1/4) * (
        mse(noisy1, pred2) + mse(noisy1, pred3) + mse(noisy1, pred4) +
        mse(noisy2, pred1) + mse(noisy2, pred3) + mse(noisy2, pred4) +
        mse(noisy3, pred1) + mse(noisy3, pred2) + mse(noisy3, pred4) +
        mse(noisy4, pred1) + mse(noisy4, pred2) + mse(noisy4, pred3)
    )

    '''

    ############### S2B ##################
    '''
    noisy1, noisy2, noisy3, noisy4 = S2B(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    pred3 = noisy3 - model(noisy3)
    pred4 = noisy4 - model(noisy4)


    # 計算損失，使用每個區塊與其他區塊去噪後結果的損失

    loss_quad = (1/4) * (
        mse(noisy1, pred2) + mse(noisy1, pred3) + mse(noisy1, pred4) +
        mse(noisy2, pred1) + mse(noisy2, pred3) + mse(noisy2, pred4) +
        mse(noisy3, pred1) + mse(noisy3, pred2) + mse(noisy3, pred4) +
        mse(noisy4, pred1) + mse(noisy4, pred2) + mse(noisy4, pred3)
    )
    loss = loss = loss_res + loss_cons + loss_quad
    '''
    return loss

def train(model, optimizer, noisy_img):

  loss = loss_func(model,noisy_img)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss.item()

def test(model, noisy_img, clean_img):

    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img),0,1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10*np.log10(1/MSE)

    return PSNR

def denoise(model, noisy_img):

    with torch.no_grad():
        pred = torch.clamp( noisy_img - model(noisy_img),0,1)

    return pred

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
class network(nn.Module):
    def __init__(self,n_chan,chan_embed=48): #中間channals數目
        super(network, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        #input #[1,3,128,128]
        x = self.act(self.conv1(x)) #[1, 48, 128, 128]
        #print(f"the shape after conv1{x.shape}")
        x = self.act(self.conv2(x)) # [1, 48, 128, 128]
        #print(f"the shape after conv2{x.shape}")
        x = self.conv3(x) #[1, 3, 128, 128]
        #print(f"the shape after conv3{x.shape}")
        return x
    def denoise(model, noisy_img):

        with torch.no_grad():
            pred = torch.clamp( noisy_img - model(noisy_img),0,1)

        return pred
if __name__=='__main__':
    set_seed(0)
    device="cuda:0"
    PSNR_Total=0
    
    # load noisy 
    noisy_image_path = f'./KodaK24/val_noisy/val_noisy_5.png'
    noisy_image = Image.open(noisy_image_path)
    transform = transforms.ToTensor()
    noisy_img = transform(noisy_image).unsqueeze(0)
    
    # load clean
    gt_image_path = f'./KodaK24/val_gt/val_gt_5.png'
    gt_image = Image.open(gt_image_path)
    clean_img = transform(gt_image).unsqueeze(0)
    
    # add to device
    n_chan = clean_img.shape[1]
    clean_img = clean_img.to(device)
    noisy_img = noisy_img.to(device)
    # model modi
    
    model_mod = Network(n_chan)
    model_mod = model_mod.to(device)
    
    #model ori
    model_ori = network(n_chan)
    model_ori = model_ori.to(device)
    ######### mod #################
    
    max_epoch = 10000     # training epochs
    lr = 0.001           # learning rate
    step_size = 1000     # number of epochs at which learning rate decays
    gamma = 0.5          # factor by which learning rate decays

    optimizer = optim.Adam(model_ori.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    with tqdm(range(max_epoch)) as pbar:
        for epoch in pbar:
            train(model_ori, optimizer, noisy_img)
            PSNR = test(model_ori, noisy_img, clean_img)
            if PSNR>29:
                break
            scheduler.step()
            pbar.set_postfix({"PSNR": f"{PSNR:.2f}"})
            
    denoised_mod_img = denoise(model_ori, noisy_img)
    denoised_mod = denoised_mod_img.cpu().squeeze(0).permute(1,2,0).numpy()
    
    denoised_image = Image.fromarray((denoised_mod * 255).astype('uint8'))
    denoised_image.save('denoised_image_origin.png')




