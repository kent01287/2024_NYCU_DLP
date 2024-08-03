
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
from torch import stack

def get_key(fp):
    filename = fp.split('/')[-1] #/DLP/filename.jpg -> filename.jpg
    filename = filename.split('.')[0].replace('frame', '') #frame123.jpg -> 123.jpg
    return int(filename) #output the number

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        # * means all of the png file
        if mode == 'train':
            self.img_folder     = sorted(glob(os.path.join(root, 'train/train_img/*.png')), key=get_key) #根據編號排序
            self.prefix = 'train'
        elif mode == 'val':
            self.img_folder     = sorted(glob(os.path.join(root, 'val/val_img/*.png')), key=get_key)
            self.prefix = 'val'
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        path = self.img_folder[index]
        
        imgs = []
        labels = []
        for i in range(self.video_len): #每次處理video_len的張數
            label_list = self.img_folder[(index*self.video_len)+i].split('/') #獲得當前偵的圖像路徑 ['img_folder', 'video1_frame1.png',...七張]
            label_list[-2] = self.prefix + '_label'#修改倒數第二偵  #['img_folder', 'video1_label.png'...七張]
            
            img_name    = self.img_folder[(index*self.video_len)+i] #當前偵的圖像名字
            label_name = '/'.join(label_list)# 'img_folder/video1_label.png'
            
            imgs.append(self.transform(imgloader(img_name)))
            labels.append(self.transform(imgloader(label_name)))
        return stack(imgs), stack(labels)
    
if __name__ == '__main__':
    transform = transforms.Compose([
            transforms.Resize((32,64)),
            transforms.ToTensor()
        ])
    dataset_train = Dataset_Dance(root='./Lab4/LAB4_Dataset/LAB4_Dataset',mode='train',transform=transform)
    dataset_val = Dataset_Dance(root='./Lab4/LAB4_Dataset/LAB4_Dataset',mode='val',transform=transform)
    
    print("Training dataset length:", len(dataset_train)) # 3344
    print("Validation dataset length:", len(dataset_val)) # 90
    
    imgs, labels = dataset_train[0]
    
    print("Shape of images:", imgs.shape) # [7,3,32,64]  7張圖片作為一個dataset
    print("Shape of labels:", labels.shape)# [7,3,32,64]
    #所以包進dataloader變成 [batch_size,video_len,channals,H,W]
    
    

