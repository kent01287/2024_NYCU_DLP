import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
from utils import LoadTestData, LoadMaskData
from torch.utils.data import Dataset,DataLoader
from torchvision import utils as vutils
import os
from models import MaskGit as VQGANTransformer
import yaml
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
'''
 VQ（矢量量化）過程：
圖像首先通過一個卷積神經網絡（CNN）等編碼器被轉換成一個低維的特徵空間（通常這一步會壓縮圖像的尺寸，例如將 256x256 的圖像壓縮為 16x16 或 32x32 的特徵圖）。
接下來，這些特徵圖中的每個區域會被量化成一個離散的編碼，這個編碼就是一個 token。這些 tokens 通常取自於一個預先訓練好的「詞彙表」（codebook），這樣不同的圖像區域會被映射成相應的 token。
'''
#VQVAE 提出了一個discrete repregentation for latent space，他需要學習，他需要學習embedding space，also known as codebook 大小(K*D) K=H*W
#一張圖片 經NN後 成為 H*W*D 的vector each vector 去codebook找最接近的vector索引 標在vector 上 變成 H*W的feature matrix
'''
VQGAN 是 VQVAE的改良版本，結合了 autoregressive transformer
改動有二:
-原本loss function MSE改成Perceptual(感知) Loss 做為reconstruction loss 也引入GAN的對抗機制 合併了patch-based Discriminator
-引入Autoregressive Transformer 用來整合上下文訊息來生成token,更好捕捉全局與局部特徵
'''
'''
MaskGit 主要遵循了VQGAN的作法，解決了VQGAN在self transformer 表現不佳的問題(單向預測來參考長序列的token，導致生成較慢)
MaskGit 透過雙向transformer進行token生成來解決這個問題，雙向模型能夠一次預測所有token
MaskGit  引入了受人類繪畫邏輯啟發的遮罩視覺 token 建模（MVTM）訓練機制。這一核心概念涉及到最初保留一部分具有高可信度的 
tokens，並逐步精細化被遮罩的 tokens。
'''
'''
Tokenization：

一張輸入的圖像首先經過編碼器（Encoder），然後通過矢量量化（VQ）步驟將圖像轉換成一個由許多視覺 tokens 組成的矩陣。這些視覺 tokens 是圖片的高級表示形式。
Masked Visual Token Modeling (MVTM)：

在訓練過程中，部分 tokens 會被隨機遮罩掉（Masked Tokens）。這些遮罩的 tokens 被替換成特定的符號，使模型在生成過程中預測這些被遮罩的位置。
然後，這些遮罩後的 tokens 作為輸入送入一個雙向 Transformer 模型中。
Bidirectional Transformer：

雙向 Transformer 的核心特點是能夠同時參考一個 tokens 序列中的所有位置，而不僅僅是前面的 tokens（這與自回歸模型不同）。
這意味著模型能夠在一次運行中同時預測所有被遮罩的 tokens，從而加快圖像生成速度。這裡的輸出是經過 Transformer 預測後的 tokens（Predicted Tokens）。
Reconstruction：

最後，預測到的完整 tokens 序列經過解碼器（Decoder），重建出最終的圖像。
核心概念解釋：
視覺 Tokens：這是圖像的高級特徵表示，經過編碼器和矢量量化步驟得到的。
雙向 Transformer：這是 MaskGIT 的核心模型，用來預測遮罩的 tokens。雙向 Transformer 可以同時考慮上下文信息（即整個 tokens 矩陣），這使得預測更加準確且高效。
遮罩視覺 Token 建模 (MVTM)：這是一種類似於遮罩語言模型（如 BERT）的技術，但用於圖像。模型通過預測遮罩的位置來學習圖像的內部結構。
總結來說，MaskGIT 的架構通過雙向 Transformer 和 MVTM 來有效提高圖像生成的速度和質量，尤其在處理多個遮罩 token 的情境下，能夠一次性預測全部遮罩區域，大大加快生成速度。
'''
class MaskGIT:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.model.load_transformer_checkpoint(args.load_transformer_ckpt_path)
        self.model.eval()
        self.total_iter=args.total_iter #decode要花幾步
        self.mask_func=args.mask_func
        self.sweet_spot=args.sweet_spot
        self.device=args.device
        self.prepare()

    @staticmethod
    def prepare():
        os.makedirs("./Results/test_results", exist_ok=True)
        os.makedirs("./Results/mask_scheduling", exist_ok=True)
        os.makedirs("./Results/imga", exist_ok=True)
        os.makedirs("Results", exist_ok=True)

##TODO3 step1-1: total iteration decoding  
#mask_b: iteration decoding initial mask, where mask_b is true means mask
#[3,64,64]
    def inpainting(self,image,mask_b,i): #MakGIT inference
        '''
        image 原始圖像(有些缺補)
        mask_b 對應遮罩
        '''
        # img[B,3,64,64] mask[B,1,16,16]
        maska = torch.zeros(self.total_iter, 3, 16, 16) #save all iterations of masks in latent domain
        imga = torch.zeros(self.total_iter+1, 3, 64, 64)#save all iterations of decoded images
        #圖像標準化
        mean = torch.tensor([0.4868, 0.4341, 0.3844],device=self.device).view(3, 1, 1) 
        std = torch.tensor([0.2620, 0.2527, 0.2543],device=self.device).view(3, 1, 1)
        ori=(image[0]*std)+mean
        #print(image[0].shape)
        imga[0]=ori #mask the first image be the ground truth of masked image

        self.model.eval()
        with torch.no_grad():
            _,z_indices = self.model.encode_to_z(image)  #z_indices: masked tokens (b,16*16)
            mask_num = mask_b.sum().item()
            z_indices=z_indices.view(-1,256) #變成(b,256)
            
            
            #z_indices_predict=z_indices
            mask_bc=mask_b  #上一階段mask的value
            mask_b=mask_b.to(device=self.device)
            mask_bc=mask_bc.to(device=self.device)
            
            #raise Exception('TODO3 step1-1!')
            ratio = 0
            #iterative decoding for loop design
            #Hint: it's better to save original mask and the updated mask by scheduling separately
            for step in range(self.total_iter):
                if step == self.sweet_spot:
                    break
                ratio = step / self.total_iter   #this should be updated
    
                z_indices_predict, mask_bc = self.model.inpainting(z_indices,ratio,mask_bc,mask_num=mask_num)

                #static method yon can modify or not, make sure your visualization results are correct
                mask_i=mask_bc.view(1, 16, 16) #[1,256] -> [1,16,16]
                mask_image = torch.ones(3, 16, 16)
                indices = torch.nonzero(mask_i, as_tuple=False)#label mask true
                mask_image[:, indices[:, 1], indices[:, 2]] = 0 #3,16,16
                maska[step]=mask_image
                shape=(1,16,16,256)
                z_q = self.model.vqgan.codebook.embedding(z_indices_predict).view(shape)
                z_q = z_q.permute(0, 3, 1, 2)
                decoded_img=self.model.vqgan.decode(z_q)
                dec_img_ori=(decoded_img[0]*std)+mean
                imga[step+1]=dec_img_ori #get decoded image

            ##decoded image of the sweet spot only, the test_results folder path will be the --predicted-path for fid score calculation
            vutils.save_image(dec_img_ori, os.path.join("./Results/test_results", f"image_{i:03d}.png"), nrow=1) 

            #demo score 
            vutils.save_image(maska, os.path.join("./Results/mask_scheduling", f"test_{i}.png"), nrow=10) 
            vutils.save_image(imga, os.path.join("./Results/imga", f"test_{i}.png"), nrow=7)



class MaskedImage:
    def __init__(self, args):
        mi_ori=LoadTestData(root= args.test_maskedimage_path, partial=args.partial)
        self.mi_ori =  DataLoader(mi_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        mask_ori =LoadMaskData(root= args.test_mask_path, partial=args.partial)
        self.mask_ori =  DataLoader(mask_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        self.device=args.device

    def get_mask_latent(self,mask):    
        downsampled1 = torch.nn.functional.avg_pool2d(mask, kernel_size=2, stride=2)
        resized_mask = torch.nn.functional.avg_pool2d(downsampled1, kernel_size=2, stride=2)
        resized_mask[resized_mask != 1] = 0       #1,3,16*16   check use  
        mask_tokens=(resized_mask[0][0]//1).flatten()   ##[256] =16*16 token
        mask_tokens=mask_tokens.unsqueeze(0)
        mask_b = torch.zeros(mask_tokens.shape, dtype=torch.bool, device=self.device)
        mask_b |= (mask_tokens == 0) #true means mask
        return mask_b


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT for Inpainting")
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')#cuda
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    
    parser.add_argument('--MaskGitConfig', type=str, default='./config/MaskGit.yml', help='Configurations for MaskGIT')
    
    
#TODO3 step1-2: modify the path, MVTM parameters
    parser.add_argument('--load-transformer-ckpt-path', type=str, default='./transformer_checkpoints/ckpt_100_Best.pt', help='load ckpt')
    #FID 43.2 with train(FID_41)60epoch 且 (sweet spot ,total iter) (5,5,consine) #別人寫的42 #44
    #FID 46 with train(FID_41)60epoch 且 (sweet spot ,total iter) (1,5,consine)
    #FID 44 with training 100epoch 且(5,5,consine)
    #FID 48 with training 100epoch 且(10,10,consine)
    #FID 43 with training 80epoch 且(5,5,consine)
    #FID 43.8 with training 60epoch 且(5,5,consine)
    #FID 47.8 with training 60epoch 且(10,10,consine)
    ################### Different opti and scheduler######################
    #FID 40.7384 with training 100epoch 且(10,10,consine)
    #FID 35.9461 with training 100epoch 且(5,5,consine)
    #FID 38.4948 with training 100epoch 且(5,5,linear)
    #FID 44.2150 with training 100epoch 且(10,10,linear)
    #FID 39.7163 with training 100epoch 且(10,10,square)
    #FID 36.0789 with training 100epoch 且(5,5,square)
    #######################D##############################
    #FID 42.57 with training 100epoch 且(5,5,cosine)
    
    #dataset path
    parser.add_argument('--test-maskedimage-path', type=str, default='./lab5_dataset/masked_image', help='Path to testing image dataset.')
    parser.add_argument('--test-mask-path', type=str, default='./lab5_dataset/mask64', help='Path to testing mask dataset.')
    #MVTM parameter
    parser.add_argument('--sweet-spot', type=int, default=5, help='sweet spot: the best step in total iteration') #希望在某個步驟進行調整
    parser.add_argument('--total-iter', type=int, default=5, help='total step for mask scheduling')
    parser.add_argument('--mask-func', type=str, default='cosine',choices=['linear','cosine','square'], help='mask scheduling function')

    args = parser.parse_args()

    t=MaskedImage(args)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    MaskGit_CONFIGS["model_param"]['gamma_type'] = args.mask_func
    maskgit = MaskGIT(args, MaskGit_CONFIGS)

    i=0
    for image, mask in tqdm(zip(t.mi_ori, t.mask_ori)):
        image=image.to(device=args.device)
        mask=mask.to(device=args.device)
        mask_b=t.get_mask_latent(mask)       
        maskgit.inpainting(image,mask_b,i)
        i+=1
        


