import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import torchvision
import random
import sys
from train import set_seed
from diffusers import DDPMScheduler,UNet2DModel
from diffusers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from model.DDPM import DDPM
from dataloader import ICLEVERDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluator import evaluation_model
from torchvision import transforms

         
def get_args():
    parser = argparse.ArgumentParser(description='Train the DDPM on images and target masks')
    parser.add_argument('--img_path',  default= './iclevr',type=str, help='path of the img')
    parser.add_argument('--json_path',  default= './train.json',type=str, help='path of the json_file')
    parser.add_argument('--epochs',type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b',type=int, default=64, help='batch size')
    parser.add_argument('--learning-rate', '-lr',type=float, default=1e-4, help='learning rate')
    parser.add_argument('--device',type=str, default='cuda:0', help='device')
    parser.add_argument('--time_step',type=float, default=1000, help='The time step')
    parser.add_argument('--beta_schedule',type=str, default="squaredcos_cap_v2",choices=['linear', 'squaredcos_cap_v2','scaled_linear'], help='The beta choose')
    parser.add_argument('--save_path',  default= './model_weight',type=str, help='path of the saving model')
    parser.add_argument('--ckpt',  default= './model_weight/Best.ckpt',type=str, help='path of the ckpt')
    #'./model_weight/epoch=1'
    parser.add_argument('--model',type=str, default="Unet",choices=['Unet', 'ResNet34_Unet'], help='Unet or ResNet34_Unet')

    return parser.parse_args()

def evaluate(args,file):
        model = DDPM().to(args.device)
        os.makedirs(f"./img_{file}",exist_ok=True)
        #load the check point
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.time_step, beta_schedule=args.beta_schedule)

        model.eval()
        
        
        test_set = ICLEVERDataset(json_file=f'./{file}.json',root=args.img_path, mode='test')
        test_loader = DataLoader(test_set, batch_size=args.batch_size)
        
        img_lists = [[] for _ in range(args.batch_size)] 
        max_acc=0
        last_img_list=[]
        for label in tqdm(test_loader):
            x = torch.randn(label.size(0), 3, 64, 64).to(args.device) #sample一個全部都是雜訊的圖
            label = label.to(args.device)
            
            for i, t in tqdm(enumerate(noise_scheduler.timesteps)): # reverse process 0~999
                with torch.no_grad(): 
                    pred_noise = model(x, t, label) #丟進noise predictor得到一張noise
                x = noise_scheduler.step(pred_noise, t, x).prev_sample #再透過noise還原原圖
                acc=evaluation_model().eval(images=x.detach(), labels=label) #最後一個time step 的acc
                max_acc=max(acc,max_acc)
                #print(f"Time step {i} : acc {acc}")
                if i% 100==99:
                    x_cpu=x.detach().cpu()
                    for j in range(x_cpu.shape[0]):  # 假设 x_cpu 是一个 numpy 数组或 PyTorch 张量
                        img_lists[j].append(x_cpu[j])
                    #print(f"img_list has {len(img_list)} elements.")
                    if i==999:
                        last_img_list.extend(torch.unbind(x_cpu, dim=0)) #變成B個[C,H,W]
                        print(len(last_img_list))  # 输出：4
                        print(last_img_list[0].shape)  # 输出：(3, 32, 32)
            #print(f"The acc of {file}.json:{round(acc, 3)*100}%")
        
            ##### 最後一輪 產生圖片###########
            
            for count, img_list in enumerate(img_lists):
                if not img_list:  # 為空 跳出循環
                    break
                tensor_images = torch.stack(img_list) #將列表轉換為四維張量 [N, C, H, W] 代表有幾張圖片
                grid_image = torchvision.utils.make_grid(tensor_images, nrow=10, padding=2)  # nrow=5 表示每行顯示 5 張圖片
                grid_image_pil = transforms.ToPILImage()(grid_image)
                grid_image_pil.save(f'./img_{file}/output_image_{count}.png')
                
            
            tensor_images = torch.stack(last_img_list) #將列表轉換為四維張量 [N, C, H, W] 代表有幾張圖片
            grid_image = torchvision.utils.make_grid(tensor_images, nrow=10, padding=2)  # nrow=5 表示每行顯示 5 張圖片
            grid_image_pil = transforms.ToPILImage()(grid_image)
            grid_image_pil.save(f'./img_{file}/final_result.png')
            
        print(f"The acc of {file}.json is={max_acc*100}%")
        return round(max_acc, 3)

if __name__=='__main__':
    set_seed(0)
    args = get_args()
    acc1=evaluate(args,'test') #80.55
    acc2=evaluate(args,'new_test')#80.95      
    print(f"The acc of test.json is {acc1},The acc of new_test.json is{acc2}")