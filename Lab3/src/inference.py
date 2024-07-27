import argparse
import torch
import torch.nn as nn
import os
import numpy as np

from PIL import Image
from models.resnet34_unet import ResNet34_UNet
from models.unet import Unet
from oxford_pet import load_dataset
from torch.utils.data import  DataLoader
from utils import dice_score

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='../saved_models/ResNet34_Unet/model_ResNet34_Unet_0.91.pth', help='path to the stored model weoght')
    parser.add_argument('--Unet', default='../saved_models/Unet/model_Unet_0.91.pth', help='Unet')
    parser.add_argument('--ResNet34_Unet', default='../saved_models/ResNet34_Unet/model_ResNet34_Unet_0.91.pth', help='ResNet34_Unet')
    parser.add_argument('--data_path', type=str, default='../dataset',help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--models',type=str, default="ResNet34_Unet",choices=['Unet','ResNet34_Unet'], help='Unet or ResNet34_Unet ')
    return parser.parse_args()

def test(args):
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #load the model
    if(args.models=='Unet'):
        print("Using Unet")
        model = Unet(in_ch=3,out_ch=1).to(device)
        model.load_state_dict(torch.load(args.model))
    elif(args.models=='ResNet34_Unet'):
        print("Using ResNet34_Unet")
        model = ResNet34_UNet(in_ch=3,out_ch=1).to(device)
        model.load_state_dict(torch.load(args.model))
        
    model.eval()
    #load the test data 
    test_data = load_dataset(data_path=args.data_path,mode='test')
    test_loader = DataLoader(test_data,batch_size=args.batch_size,shuffle=True,pin_memory=True,num_workers=16)

    result=0
    
    with torch.no_grad():
        for sample in test_loader:
            image, mask = sample["image"].to(device), sample["mask"].to(device)
            pred = model(image.to(dtype=torch.float32))
            #output = model(image.to(dtype=torch.float32))
            pred=torch.sigmoid(pred)
            ######## calculate dice score##########
            result += dice_score(pred,mask)
            
        score= result / len(test_loader)
        print(f"The dice score is {score:.2f}")

    return score

# show the origin and pred picture
def show(args): # plot the grapg with origin and not origin
    #make the dirs to put the image
    os.makedirs("../img",exist_ok=True)
    # "cuda" only when GPUs are available.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_data = load_dataset(data_path=args.data_path,mode='test')
    test_loader = DataLoader(test_data,batch_size=args.batch_size,shuffle=True,pin_memory=True)
    
    data_iter = iter(test_loader)  
    batch = next(data_iter)  
    image, mask = batch["image"], batch["mask"]
    #convert to numpy
    image_np = image.numpy()
    # eliminate the first dim
    image_np = np.squeeze(image_np, axis=0)
    # change CHW -> HWC
    image_np = np.moveaxis(image_np, 0, 2) 

    image_np = image_np.astype(np.uint8)
    
    original_image = Image.fromarray(image_np)
    original_image.save(f'../img/original_image.png')
    
    ######################## mask result #################################
    #load the model
    
    
    model1 = Unet(in_ch=3,out_ch=1).to(device)
    model1.load_state_dict(torch.load(args.Unet))
   
    model2 = ResNet34_UNet(in_ch=3,out_ch=1).to(device)
    model2.load_state_dict(torch.load(args.ResNet34_Unet))
    
    ############### predict the result (Unet)  ###################################
    image, mask = batch["image"].to(device), batch["mask"].to(device)
    with torch.no_grad():
        pred = model1(image.to(dtype=torch.float32))
        pred = torch.sigmoid(pred) #map to zero or one
    
    pred = (pred > 0.5).float() 
    pred_np = pred.cpu().numpy()
    ######### handle dim####################
    #print(pred_np.shape) #(1,65536)
    pred_np=pred_np.reshape(1,1,256,256) #(1,1,256,256)
    pred_np = np.squeeze(pred_np, axis=0) #(1,256,256)
    pred_np = np.moveaxis(pred_np, 0, 2) #(256,256,1)
    pred_np = pred_np.astype(np.uint8)
    pred_np = np.squeeze(pred_np, axis=2)# (256,256)
    ####### coloring ##########
   
    green = [0, 255, 0]  # greeb
    blue = [0, 0, 255]   # blue
    color_image = np.zeros((256, 256, 3), dtype=np.uint8)
    color_image[pred_np ==1] = green  # <0.5 -> green
    color_image[pred_np == 0] = blue  # >=0.5 blue

    #print(type(pred_np))
    original_image = Image.fromarray(color_image)
    original_image.save(f'../img/pred_image_Unet.png')
    
    ############### predict the result (ResNet34)  ###################################
    image, mask = batch["image"].to(device), batch["mask"].to(device)
    with torch.no_grad():
        pred = model2(image.to(dtype=torch.float32))
        pred = torch.sigmoid(pred) #map to zero or one
    
    pred = (pred > 0.5).float() 
    pred_np = pred.cpu().numpy()
    ######### handle dim####################
    #print(pred_np.shape) #(1,65536)
    pred_np=pred_np.reshape(1,1,256,256) #(1,1,256,256)
    pred_np = np.squeeze(pred_np, axis=0) #(1,256,256)
    pred_np = np.moveaxis(pred_np, 0, 2) #(256,256,1)
    pred_np = pred_np.astype(np.uint8)
    pred_np = np.squeeze(pred_np, axis=2)# (256,256)
    #print(type(pred_np))
    original_image = Image.fromarray(color_image)
    original_image.save(f'../img/pred_image_ResNet34_Unet.png')
    
#combine the origin and mask image
def combine(pic1,pic2): # merge the two picture
    #conver the image to RGB
    image1 = Image.open(pic1).convert('RGB')
    image2 = Image.open(pic2).convert('RGB')
    img =Image.blend(image1, image2, alpha=0.5)
    # Extract the base name of the second image file
    pic2_basename = os.path.basename(pic2)
    
    # Determine the output file name based on pic2's name
    if 'pred_image_ResNet34_Unet.png' in pic2_basename:
        output_filename = 'blend_image_ResNet34_Unet.png'
    elif 'pred_image_Unet.png' in pic2_basename:
        output_filename = 'blend_image_Unet.png'
    else:
        output_filename = 'blend_image.png'  # Default name if no match
    
    # Save the blended image with the determined name
    output_path = os.path.join('../img', output_filename)
    img.save(output_path)
    print(f"Combined image saved as {output_path}")
   
if __name__ == '__main__':
    args = get_args()
    #test(args)
    show(args)
    combine('../img/original_image.png','../img/pred_image_Unet.png')
    combine('../img/original_image.png','../img/pred_image_ResNet34_Unet.png')
    #assert False, "Not implemented yet!"