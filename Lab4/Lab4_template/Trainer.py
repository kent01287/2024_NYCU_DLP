import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10
'''
在變分自編碼器VAE中，KL_annealing 是一種技術，用於控制模型的KL散度（Kullback-Leibler divergence）的貢獻，
通常在訓練過程中逐步增加其權重。這樣做的目的是為了在訓練初期避免KL散度對損失函數的影響過大，
並在後期逐漸引入KL散度，使模型能夠更好地學習潛在空間的結構。
在KL退火過程中，beta 是一個參數，用來調整KL散度的權重。beta 在訓練初期通常設置為一個較小的值，隨著訓練的進行，beta 逐漸增加，直到達到預定的值。

beta 的值：在VAE的損失函數中，beta 用來調整KL散度的影響。
如果 beta 設為1，則KL散度的貢獻與原始損失函數中的設置相同。
如果 beta 大於1，則KL散度的貢獻被放大，強調模型學習潛在變數分佈接近標準正態分佈的需求。


VAE的loss 由 reconstruction loss和 KL divergence組成
reconstruction loss 從潛在變量生成的數據與原始數據之間的差異。常見的重建損失函數是MSE
KL divergence
'''
#PSNR（峰值信噪比）是一種用來衡量圖像或視頻壓縮後質量的指標
#PSNR 高 ->圖像趨近原始 低 -> 不趨近原始 >30算還不錯了
def Generate_PSNR(imgs1, imgs2, data_range=1.): #經過標準化後的圖片
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    #若batch size >1 是計算平均每個batch_size 的mse 所以會錯誤
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

#KL-divergence 來衡量兩個distribution的差異
#從近似後驗分佈 q(z|x) 到到先驗分佈 p(z)的差距  希望 min encoder的部分 
def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD

#KL (Kullback-Leibler) annealing 
# 是一種在訓練VAE時，逐步增大KL散度項權重的方法。其目的是為了平衡Reconstruction loss和KL divergence之間的權重，
#lower bound = reconstruction loss + KL (q(z|x)||p(z))
#L=Reconstruction Loss+β⋅KL Divergence
#KL annealing 的過程可以理解為逐步增大β的值，使得模型在訓練初期更注重Reconstruction loss，
class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.args=args
        self.current_epoch=current_epoch
        self.type = args.kl_anneal_type #linear step cyclic
        self.cycle = args.kl_anneal_cycle #變化週期
        self.ratio = args.kl_anneal_ratio # 變化的大小 >1 需要更多時間到達最終值
        self.total_epoch = args.num_epoch
        if self.type== 'None':
            self.beta=1.0
        else:
            self.beta=0.0
        #raise NotImplementedError
        
    def update(self):
        # TODO
        self.current_epoch+=1
        #self.frange_cycle_linear(n_iter = self.total_epoch,n_cycle=self.cycle , ratio=self.ratio)
        
        self.frange_cycle_linear(n_iter = self.current_epoch,n_cycle=self.cycle , ratio=self.ratio)
        #raise NotImplementedError
    
    def get_beta(self):
        return self.beta
    
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=10, ratio=1):
        tmp = 0
        tmp = ((n_iter % n_cycle) / n_cycle) * ratio #mapping 到 0 1 之間
        if(tmp <= stop-0.1):
            self.beta = tmp
        else:
            self.beta = stop        
            
        if(self.type=='Monotonic' and self.n_iter>=n_cycle): #若monotonic 後期一直設為1
            self.beta=stop
        
        
        
class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        #F_dim = 128
        self.frame_transformation = RGB_Encoder(3, args.F_dim) #(3,128)
        self.label_transformation = Label_Encoder(3, args.L_dim) #(3,32)
        
        # Conduct Posterior prediction in Encoder
        # Posterior q(z|x)
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim) #N_dim 代表noise的dim (128+32,12)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)#(128+32+12,192)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len #16
        self.val_vi_len   = args.val_vi_len #630
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        train_loss_list=[]
        val_loss_list=[]
        PSNR_list=[]
        tfr_list=[] 
        kl_list=[]
        print(args.num_epoch)
        
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False #是否使用Teacher forcing
            epoch_loss=0
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing) #reconstruction loss + KL divergence
                epoch_loss+=loss.item()
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    #detach() 方法會創建一個新的張量，這個張量與原始損失值共享數據，但不會參與反向傳播，即不計算梯度
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            #每隔一定的 epoch（self.args.per_save）保存一次模型的檢查點到指定的路徑
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
                
            val_loss,PSNR = self.eval()
            
            epoch_loss=epoch_loss/len(train_loader)
            train_loss_list.append(epoch_loss)
            tfr_list.append(self.tfr)
            kl_list.append(beta)
            
            val_loss_list.append(val_loss.numpy())
            PSNR_list.append(PSNR.numpy())
            
            print(f"epoch:{i},train_losee{epoch_loss:.4f},val_loss:{val_loss:.4f},PSNR:{PSNR}")
            
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
        
        
        np.save(f"./Lab4/img/train_losses.npy", np.array(train_loss_list))
        np.save(f"./Lab4/img/val_losses.npy", np.array(val_loss_list))
        np.save(f"./Lab4/img/PSNR.npy",np.array(PSNR_list))
        
        
        np.save(f"./Lab4/img/tfr.npy", np.array(tfr_list))
        np.save(f"./Lab4/img/kl.npy", np.array(kl_list)) 
         
    @torch.no_grad() #函數內部不會計算gradient
    def eval(self):
        val_loader = self.val_dataloader()
        total_images=0
        val_PSNR=0
        val_loss=0
        #tqdm(val_loader, ncols=120) 賦值給pbar
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            total_images += img.size(0)  # 累計每個批次的圖像數量
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss ,PSNR = self.val_one_step(img, label)
            val_PSNR+=PSNR
            val_loss+=loss
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        
        return val_loss/len(val_loader),val_PSNR/len(val_loader)
    
    def training_one_step(self, img, label, adapt_TeacherForcing): #[batch,時間序列,channals,height,width] 
        #Define initital value
        KL=0
        Reconstruction_loss=0
        
        #permute 成 [時間序列,batch,channals,height,width]
        img=img.permute(1,0,2,3,4)
        label = label.permute(1,0,2,3,4)
        out = img[0]
        
        for i in range(1,self.train_vi_len-1): #處理dataset裡面的一個data 因為包含多個時間序列 (16)
            if adapt_TeacherForcing: #用前一偵的結果
                img_out = img[i-1]
            else:
                img_out = img[i-1] * self.tfr + out * (1 - self.tfr)
     
            img_out=self.frame_transformation(img_out) #imge_out [batch,c,h,w]
            label_out=self.label_transformation(label[i])
            #pass encoder
            z,mu,logvar = self.Gaussian_Predictor(img_out,label_out)
            #pass to decoder
            param=self.Decoder_Fusion(img_out,label_out,z)
            out = self.Generator(param)
            #calculate loss
            Reconstruction_loss += self.mse_criterion(out, img[i])
            KL += kl_criterion(mu, logvar, batch_size = self.batch_size)
            
        loss = Reconstruction_loss + self.kl_annealing.get_beta() * KL
        
        self.optim.zero_grad()
        loss.backward()
        # Gradient clipping
        #max_norm = 1.0  # You can adjust this value
        #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        self.optimizer_step()
        
        return loss.detach()
        
    
    def val_one_step(self, img, label):
        KL=0.0
        PSNR_total=0.0
        Reconstruction_loss=0.0
        #permute 成 [時間序列,batch,channals,height,width]
        img=img.permute(1,0,2,3,4)
        label = label.permute(1,0,2,3,4)
        psnr_frame=[]
        for i in range(1,self.val_vi_len): #處理dataset裡面的一個data 因為包含多個時間序列 (630)
    
            img_out=self.frame_transformation(img[i]) #imge_out [batch,c,h,w]
            label_out=self.label_transformation(label[i])
            
            #pass encoder
            #z,mu,logvar = self.Gaussian_Predictor(img_out,label_out)
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()#從normal distribtion smaple一個出來
            param=self.Decoder_Fusion(img_out,label_out,z)
    
            out = self.Generator(param)
            
            #calculate loss
            Reconstruction_loss += self.mse_criterion(out, img[i])
            #KL += kl_criterion(mu, logvar, batch_size = self.batch_size)
            
            #calculate PSNR
            PSNR= Generate_PSNR(out,img[i])
            psnr_frame.append(PSNR.detach().cpu().numpy())
            PSNR_total+=PSNR
            
        loss=Reconstruction_loss
        psnr=PSNR_total/self.val_vi_len
        return loss.cpu().detach(),psnr.cpu().detach()
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0) #每偵40毫秒 無限loop
    
    def train_dataloader(self):
        #do some DA
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        #DR : data_path of dataset
        #training video len
        #fast_partial : Use part of the training data to fasten the convergence
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    #tfr 代表使用ground truth 的機率 越大代表使用機率越高
    #我們傾向訓練初期使用高一點的tfr 這樣模型比較不會出錯
    #後期我們使用比較低的tfr 讓模型可以自己生成
    
    def teacher_forcing_ratio_update(self):
        # TODO
        #tfr_sde:The epoch that teacher forcing ratio start to decay
        #tfr_d_step : Decay step that teacher forcing ratio adopted (0.1)
        if self.current_epoch >= self.tfr_sde and self.tfr > 0:
            # start decay
            self.tfr-=self.tfr_d_step
        self.tfr=max(0,self.tfr)
            
        #raise NotImplementedError
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=f"{loss:.4f}", refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

def plot_loss(np1,np2,save_path=None): 
    data1 = np.load(np1)
    data2 = np.load(np2)

    
    plt.figure()

    plt.plot(data1, label='train_loss')
    plt.plot(data2, label='val_loss')
    plt.title('loss curve')
    
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='jpg')  # Save as JPG with high quality
    plt.show()
    plt.close()
def plot_beta(np1,save_path=None): #
    data1 = np.load(np1)

    
    plt.figure()

    plt.plot(data1, label='Cyclic')
    plt.title("KL_aneeling beta")
    
    plt.xlabel("epochs")
    plt.ylabel("beta")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='jpg')  # Save as JPG with high quality
def plot_tfr(np1,save_path=None): #
    data1 = np.load(np1)

    
    plt.figure()

    plt.plot(data1, label='tfr')
    plt.title("Teacher forcing rate")
    
    plt.xlabel("epochs")
    plt.ylabel("tfr")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='jpg')  # Save as JPG with high quality
    plt.show()
    plt.close()
    
def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str,  default='./Lab4/LAB4_Dataset/LAB4_Dataset', help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, default='./Lab4/model'  ,help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=1,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default='./Lab4/model/epoch=99.ckpt',help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    #main(args)
    #plot_graph_single(np1='./Lab4/img/tfr.npy',label='tfr',title='tfr',topic='tfr',save_path='./Lab4/img/tfr.jpg')
    plot_loss(np1='./Lab4/img/train_losses.npy',np2='./Lab4/img/val_losses.npy',save_path='./Lab4/img/loss_curve.jpg')
