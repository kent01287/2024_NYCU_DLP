import numpy as np
import os
import torch
from torch.utils.data import  Dataset

class MIBCI2aDataset(Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        allFeatures = []
        for file in os.listdir(filePath):
            if file.endswith('.npy'):
                features = np.load(os.path.join(filePath, file))
                allFeatures.append(features)
        features = np.concatenate(allFeatures,axis=0)
        return  torch.tensor(features, dtype=torch.float32)
    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        allLabels = []
        for file in os.listdir(filePath):
            if file.endswith('.npy'):
                labels = np.load(os.path.join(filePath, file))
                allLabels.append(labels)
        labels = np.concatenate(allLabels,axis=0)
        return torch.tensor(labels, dtype=torch.int64)
        

    def __init__(self, mode , method):
        # remember to change the file path according to different experiments
        assert mode in ['train', 'test', 'finetune']
        assert method in ['SD','LOSO']
        self.mode=mode
        if mode == 'train':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            if method == 'SD':
                self.features = self._getFeatures(filePath='./dataset/SD_train/features/')
                self.labels = self._getLabels(filePath='./dataset/SD_train/labels/')
            if method == 'LOSO':
                self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
                self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
        if mode == 'finetune':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath='./dataset/FT/features/')
            self.labels = self._getLabels(filePath='./dataset/FT/labels/')
        if mode == 'test':
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            if method == 'SD':
                self.features = self._getFeatures(filePath='./dataset/SD_test/features/')
                self.labels = self._getLabels(filePath='./dataset/SD_test/labels/')
            if method == 'LOSO':
                self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
                self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')

    def __len__(self):
        return len(self.features)
        

    def __getitem__(self, idx):
        

        features =self.features[idx]
        labels = self.labels[idx]

        if self.mode == 'train' :
            features = self.normalize(features)
            
        return features ,labels
    
    
    
    
    def normalize(self, feature):
        mean = torch.mean(feature, dim=1, keepdim=True)
        std = torch.std(feature, dim=1, keepdim=True)
        feature = (feature - mean) / (std + 1e-7)  # Adding a small value to avoid division by zero   
        return feature

    
if __name__ == '__main__': 
    dataset = MIBCI2aDataset(mode='train',method='SD')
    print(len(dataset))
    #print(dataset[50])    
