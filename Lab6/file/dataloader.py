import torch
import json
import os
import numpy as np

from torch.utils.data import  Dataset
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import torchvision.transforms as transforms
'''
 implement the multilabel encoding : 24個label
'''

class ICLEVERDataset(Dataset):
    def __init__(self,root,json_file,mode):
        '''
            root : Data path to img foler
            json_file : Data path to json file (train,test,new_test)
            mode : ["train","test"]
            
        '''
        assert mode in {"train","test"}
        self.mode=mode
        self.root=root #the image root
        
        self.transform=self.transform()
        
        with open('./objects.json', 'r') as file:
            object = json.load(file)
        self.obj=list(object.keys())
        
        self.mlb = MultiLabelBinarizer(classes=self.obj)
        self.mlb.fit([self.obj])
        
        if(self.mode=="train"):
            self.img,self.label=self.load_train(json_file)
        elif(self.mode=="test"):
            self.label=self.load_test(json_file)
        
    def __len__(self):
        return len(self.label)
        

    def __getitem__(self,idx):
        if self.mode=='train':
            path=os.path.join(self.root,self.img[idx])
    
            image = Image.open(path).convert('RGB')
            image=self.transform(image)
            
            label = torch.tensor(self.label[idx], dtype=torch.float32)
            return image,label
        else:
            label = torch.tensor(self.label[idx], dtype=torch.float32)
            return label
      
        
    def load_train(self,json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        image_list = list(data.keys())
        labels_list = list(data.values())
        #change labels to one hot vector
        one_hot_encoded = self.mlb.transform(labels_list)
        #print(f"[Train]:Number of classes: {n_classes}") # this should be 24
        return image_list,one_hot_encoded
        
    
    def load_test(self,json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        one_hot_encoded = self.mlb.transform(data)
        return one_hot_encoded 
    def transform(self):
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
        return transform
        

if __name__=='__main__':
    train_dataset=ICLEVERDataset(json_file='./train.json',root='./iclevr', mode='train')
    test_dataset=ICLEVERDataset(json_file='./test.json',root='./iclevr', mode='test')
    new_test_dataset=ICLEVERDataset(json_file='./new_test.json',root='./iclevr', mode='test')
    '''
    print(len(train_dataset)) #18009
    img,labels=train_dataset[1]
    print(img.shape) #[3,64,64]
    print(labels.shape)#[24]
    
    print(len(test_dataset)) #72
    labels = test_dataset[0]
    print(labels.shape) #[24]
    
    print(len(new_test_dataset))#87
    labels=test_dataset[0]
    print(labels)#[24]
    '''
    labels=test_dataset[1]#6 #14
    print(labels.shape)
    