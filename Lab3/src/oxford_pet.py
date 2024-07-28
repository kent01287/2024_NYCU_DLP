import os
import torch
import shutil
import numpy as np
import torchvision.transforms as transforms
import random

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve




class OxfordPetDataset(torch.utils.data.Dataset): # type: ignore
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root
                                             , "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits #test ["Abyssinian_201", "Abyssinian_202"]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask)
        
        # Print before transformation
        '''
        print("Before transformation:")
        print(f"Image: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Trimap shape: {sample['trimap'].shape}")
        '''
        if self.transform is not None:
            #print("doning")
            sample = self.transform(**sample)
        '''
        # Print before transformation
        print("Before transformation:")
        print(f"Image: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Trimap shape: {sample['trimap'].shape}")
        '''
        return sample
    '''
        "image" : 3-dim np array
        "trimap" : 1-dim np array value (1,2,3)
        "mask" : 1-dim np array value (0,1)
    '''
    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0 #foreground
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0 #background and unceartain
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs): #改寫parent getitem

        sample = super().__getitem__(*args, **kwargs) #dict
        #這邊sample 改成包含image and mask的dic
        
        #print(f"Transform applied: {self.transform}")
        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)) # type: ignore
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)) # type: ignore
        #trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST)) # type: ignore

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0).flatten() #(1,H,W)
        #sample["trimap"] = np.expand_dims(trimap, 0) #(1,H,W)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def trans(**sample):
    
    #transfer to Image
    img = Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR) # type: ignore
    mask = Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST) # type: ignore
    
    ###### rotation the picture ###############

    rndn = np.random.random()
    if rndn<0.15:
        transform1=transforms.RandomHorizontalFlip(p=1)
    elif rndn < 0.3:
        transform1=transforms.RandomVerticalFlip(p=1)
    elif rndn<0.45 :
        transform1=transforms.RandomRotation([90,90])
    elif rndn<0.6 :
        transform1=transforms.RandomRotation([270,270])
    else:
        transform1=None
    
    # do DA
    if transform1 is not None:
        img=transform1(img)
        mask=transform1(mask) 
        
    ##########crop the image################
    option=[transforms.CenterCrop(size=(200,200)),None]
    
    index = random.randint(0, len(option) - 1)
    transform2=option[index]
        
    if transform2 is not None:
        img=transform2(img)
        mask=transform2(mask) 
    
    # transfer to np
    sample = dict(image= np.array(img), mask= np.array(mask))
    #print("doing trans")
    return sample


def load_dataset(data_path, mode):
    # implement the load dataset function here
     # Define data augmentation transformations
    if os.listdir(data_path) == []:
        print("Downloading the graph")
        OxfordPetDataset.download(data_path)
    if mode == 'train':
        dataset = SimpleOxfordPetDataset(root=data_path, mode=mode,transform=trans)
    else:
        dataset= SimpleOxfordPetDataset(root=data_path, mode=mode)
    return dataset
    assert False, "Not implemented yet!"
    


if __name__ == '__main__':
    dataset=load_dataset("../dataset/oxford-iiit-pet",mode='train')
    print(dataset[0])