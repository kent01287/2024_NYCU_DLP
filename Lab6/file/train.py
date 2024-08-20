import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import sys
import random

from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from model.DDPM import DDPM
from dataloader import ICLEVERDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluator import evaluation_model

'''
    implement 1 : Noise Schedule 每個時間點增加了多少噪音  [linear cosine]
    implement 2 : Time Embeddings 使用位置編碼或可學習的嵌入將時間步長轉換為特徵向量。
    class embedded 是linear layer 與圖像數據 time step一起丟入訓練
    "DownBlock2D",  # a regular ResNet downsampling block
    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    "UpBlock2D",  # a regular ResNet upsampling block
'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
def get_args():
    parser = argparse.ArgumentParser(description='Train the DDPM on images and target masks')
    parser.add_argument('--img_path',  default= './iclevr',type=str, help='path of the img')
    parser.add_argument('--json_path',  default= './train.json',type=str, help='path of the json_file')
    parser.add_argument('--epochs',type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b',type=int, default=64, help='batch size')
    parser.add_argument('--learning-rate', '-lr',type=float, default=1e-4, help='learning rate')
    parser.add_argument('--device',type=str, default='cuda:0', help='device')
    parser.add_argument('--time_step',type=float, default=1000, help='The time step')
    parser.add_argument('--beta_schedule',type=str, default="linear",choices=['linear', 'squaredcos_cap_v2','scaled_linear'], help='The beta choose')
    parser.add_argument('--save_path',  default= './model_weight_linear',type=str, help='path of the saving model')
    parser.add_argument('--ckpt',  default= './model_weight_linear/epoch=55.ckpt',type=str, help='path of the ckpt')
    #'./model_weight/epoch=1'
    parser.add_argument('--model',type=str, default="Unet",choices=['Unet', 'ResNet34_Unet'], help='Unet or ResNet34_Unet')
    


    return parser.parse_args()
'''
unet_model = UNet2DModel(
        sample_size=64,  
        in_channels=3,   
        out_channels=3,  
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256, 512),
        #norm_num_groups=32,
        down_block_types=("DownBlock2D", "DownBlock2D","DownBlock2D" ,"DownBlock2D","AttnDownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D","UpBlock2D","UpBlock2D", "AttnUpBlock2D"),
    )
'''

def evaluate(epoch,file,noise_scheduler,model):
        test_set = ICLEVERDataset(json_file=f'./{file}.json',root=args.img_path, mode='test')
        test_loader = DataLoader(test_set, batch_size=args.batch_size)
        
        max_acc=0
        model.eval()
        for label in tqdm(test_loader):
            x = torch.randn(label.size(0), 3, 64, 64).to(args.device) #sample一個全部都是雜訊的圖
            label = label.to(args.device)
            for i, t in tqdm(enumerate(noise_scheduler.timesteps)): # reverse process
                with torch.no_grad(): 
                    pred_noise = model(x, t, label) #丟進noise predictor得到一張noise
                x = noise_scheduler.step(pred_noise, t, x).prev_sample #再透過noise還原原圖
                acc=evaluation_model().eval(images=x.detach(), labels=label)
                max_acc= max(acc,max_acc)
                #print(f"Time step{i}: acc :{acc}")
                #print(f"Time step {i}: acc: {acc}")
                
        print(f"The acc of {file}.json on epoch {epoch} : {round(max_acc, 4)*100}%" )
        return round(max_acc, 4)
    
def main(args):
    cur_epoch=0
    os.makedirs(args.save_path,exist_ok=True)
    # Define the model
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.time_step, beta_schedule=args.beta_schedule) #可以選 [linear,squaredcos_cap_v2(cosine),scaled_linear]
    model = DDPM().to(args.device)
    #Define the dataset
    train_set=ICLEVERDataset(json_file=args.json_path,root=args.img_path, mode='train')
    train_loader = DataLoader(train_set, batch_size =args.batch_size, shuffle = True)
    #Define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=len(train_loader)*args.epochs,
                num_cycles=50
            )
    

    loss_list = []
    lr_list=[]
    acc1_list=[]
    acc2_list=[]
    ######### load the model ###############################
    if args.ckpt !=None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['epoch']
        loss_list = checkpoint['loss_list']
        lr_list = checkpoint['lr_list']
        acc1_list = checkpoint['acc1_list']
        acc2_list = checkpoint['acc2_list']

    print('Start_Training')
    for epoch in range(cur_epoch+1, args.epochs):
        train_loss = 0
        for img, label in tqdm(train_loader):

            img, label = img.to(args.device), label.to(args.device)
        
            noise = torch.randn_like(img) #sample noise
            #對不同的time steps 進行採樣
            timesteps = torch.randint(0, args.time_step-1, (img.shape[0],)).long().to(args.device)
    
            noisy_img = noise_scheduler.add_noise(img, noise, timesteps)

            output = model(noisy_img, timesteps, label)

            loss = loss_function(output, noise)
            train_loss+=loss
            #BP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        lr_list.append(scheduler.get_last_lr()[0])
        scheduler.step()
        epoch_loss = train_loss/len(train_loader)
        loss_list.append(epoch_loss.detach().cpu().numpy())
        print(f'Epoch {epoch}: loss = {epoch_loss:.4f} : learnging_rate={scheduler.get_last_lr()[0]}')
        
        
        
        #每三輪存
        if epoch %3 ==0:
            path=os.path.join(args.save_path, f"epoch={epoch}.ckpt")
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),  
                "scheduler_state_dict": scheduler.state_dict(),
                "lr"        :scheduler.get_last_lr()[0],
                "epoch": epoch,
                "loss_list" : loss_list,
                "lr_list" : lr_list,
                "acc1_list" : acc1_list,
                "acc2_list" : acc2_list,
            }, path)
            print(f"save ckpt to {path}")
            
            
        #每五輪算分數
        if epoch%5==0:
            acc1=evaluate(epoch,'test',noise_scheduler,model)
            acc2=evaluate(epoch,'new_test',noise_scheduler,model)
            print(f"test:{acc1*100}%,new_test:{acc2*100}%")
            acc1_list.append((cur_epoch+1,acc1))
            acc2_list.append((cur_epoch+1,acc2))
                
                
    np.save(f"./numpy_file/loss.npy", np.array(loss_list))
    np.save(f"./numpy_file/lr.npy", np.array(lr_list))
                
    np.save(f"./numpy_file/acc1.npy", np.array(acc1_list))
    np.save(f"./numpy_file/acc2.npy", np.array(acc2_list))
            
if __name__=='__main__':
    args = get_args()
    set_seed(0)
    main(args)
    
    '''
    epoch=1
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.time_step, beta_schedule=args.beta_schedule) #可以選 [linear,squaredcos_cap_v2(cosine),scaled_linear]
    model = DDPM(noise_scheduler).to("cuda")
    checkpoint = torch.load('./model_weight/epoch=1.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    evaluate(epoch,'test',noise_scheduler,model)
    '''
    
    '''
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.time_step, beta_schedule=args.beta_schedule)
    model = DDPM().to("cuda")
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['state_dict'])
   
    evaluate(61,'test',noise_scheduler,model)
    #evaluate(61,'new_test',noise_scheduler,model)
    '''
    
    
    

