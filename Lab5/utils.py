from torch.utils.data import Dataset as torchData
from glob import glob
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
class LoadTrainData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([
                transforms.ToTensor(), #Convert to tensor
                transforms.Normalize(mean=[0.4816, 0.4324, 0.3845],std=[0.2602, 0.2518, 0.2537]),# Normalize the pixel values
        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        #self.folder = glob(os.path.join(root + '/*.png'))
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Training Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
class LoadTestData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.4868, 0.4341, 0.3844],std=[0.2620, 0.2527, 0.2543]),
                        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Testing Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
class LoadMaskData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Mask Data For Inpainting Task: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
def plot_loss(np1,np2,save_path=None): 
    data1 = np.load(np1)
    data2 = np.load(np2)

    
    plt.figure()

    plt.plot(data1,label='train_loss')
    plt.plot(data2,label='val_loss')
    plt.title('loss curve')
    #plt.yscale('log') 
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path, format='jpg')  # Save as JPG with high quality
    plt.show()
    plt.close()
    
if __name__ == '__main__': 
    '''
    train_data=LoadTrainData(root='./Lab5/lab5_dataset/train')
    print(len(train_data)) #12000
    val_data=LoadTestData(root='./Lab5/lab5_dataset/val')
    print(len(val_data)) #3000
    mask_data=LoadTestData(root='./Lab5/lab5_dataset/masked_image')
    print(len(mask_data))#747
    mask64_data=LoadTestData(root='./Lab5/lab5_dataset/mask64')
    print(len(mask64_data))#747
    
    img=train_data[0]
    print(img.shape) #[3,64,64]
    '''
    plot_loss('./img/train_loss.npy','./img/val_loss.npy','./img/loss_curve.jpg')
    