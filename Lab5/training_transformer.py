import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

def tqdm_bar(pbar, desp, loss):
    pbar.set_description(desp , refresh=False)
    pbar.set_postfix(loss=float(loss), refresh=False)
    pbar.refresh()
'''
1.經過VQGAN encoder 得到 對應的codebook_mapping和z_indices (B*16*16) #of image token 即16*16
2.將這個z_indices用上mask 丟去給transformer預測得到 logits [B,num_tokens,num_codeBlockVector+1] (B,256,1025)
3.將z_indices與 logits 做cross entrope即可得到loss
'''
#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self,train_loader,epoch):
        train_loss=0
        criterion = nn.CrossEntropyLoss()
        for img in (pbar := tqdm(train_loader, ncols=120)):
            img = img.to(args.device)
            #logits [B,256,1025] z_ind:[B,256]
            ratio = np.random.rand()
            logits,z_ind=self.model(img,ratio) #解碼出logits以及對應index
            
            #calculate loss
            #[B*256,1025] vs [B*256]
            #loss = criterion(logits.view(-1, logits.size(-1)), z_ind.view(-1))
            loss = criterion(logits.reshape(-1, logits.size(-1)), z_ind.reshape(-1))
            
            self.tqdm_bar(f'Train:{epoch}',pbar,loss.detach().cpu(),self.scheduler.get_last_lr()[0])
             # optimize
            self.optim.zero_grad()
            loss.backward() 
            self.optim.step()
            
            train_loss+=loss.detach().cpu()
        
        train_loss/=len(train_loader)  
        self.scheduler.step(loss)
        return train_loss
            
            
        pass
    @torch.no_grad()
    def eval_one_epoch(self,val_loader,epoch):
        val_loss=0
        train_loss=[]
        for img in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(args.device)
            #logits [B,256,1025] z_ind:[B,256]
            ratio = np.random.rand()
            logits,z_ind=self.model(img,ratio) #解碼出logits以及對應index
            
            #calculate loss
            #[B*256,1025] vs [B*256]
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, logits.size(-1)), z_ind.view(-1))
            
            val_loss+=loss.detach().cpu()
            self.tqdm_bar(f'Val:{epoch}', pbar, loss.detach().cpu(), self.scheduler.get_last_lr()[0])
        val_loss/=len(val_loader)  
        
        return val_loss
    def tqdm_bar(self,epoch,pbar, loss, lr):
        pbar.set_description(f"{epoch}/{args.epochs}, lr:{lr}", refresh=False)
        pbar.set_postfix(loss=f"{loss:.4f}", refresh=False)
        pbar.refresh()  
          
    def lr_lambda(self,steps):
            return min((steps+1)/args.warmup_steps, 1)
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate ,betas=(0.9, 0.96),weight_decay=1e-5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.96))
        '''
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                 mode='min',  
                                                 factor=0.1,   
                                                 patience=2,  # 2個epoch沒有降低 就調降
                                                 )
        '''
        

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=self.lr_lambda)
        #let the lr increase gradually
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=20, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')#1e-4
    parser.add_argument('--warmup-steps', type=int, default=60, help='Warmup steps.')

    parser.add_argument('--MaskGitConfig', type=str, default='./config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5: 
    train_loss_list=[]
    val_loss_list=[]
    os.makedirs("img", exist_ok=True)
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        ##### Training #############
        train_transformer.model.train()
        train_loss=train_transformer.train_one_epoch(train_loader,epoch)
        train_loss_list.append(train_loss.detach().cpu())
        ###### Val #############
        train_transformer.model.eval()
        with torch.no_grad():
            val_loss = train_transformer.eval_one_epoch(val_loader,epoch)
            val_loss_list.append(val_loss.detach().cpu())
        
        print(f'Epoch[{epoch}/{args.epochs}],Train_loss:{train_loss:.4f},Val_loss:{val_loss:.4f}')
        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.transformer.state_dict(), f"./transformer_checkpoints/ckpt_{epoch}.pt")
            print(f"Model saved at epoch {epoch}")
    np.save(f"./img/train_loss.npy", np.array(train_loss_list))
    np.save(f"./img/val_loss.npy", np.array(val_loss_list))
            