import argparse
import torch
import torch.nn as nn
import os
import numpy as np

from models.unet import Unet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from torch.utils.data import  DataLoader
from utils import dice_score 
from tqdm import tqdm


def train(args,model,save_path):
    # implement the training function here
    
    # make the dir to save different model
    os.makedirs(f"{save_path}/{args.model}", exist_ok=True)
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #Define model
    model = model.to(device)
    #Prepare Data
    train_data = load_dataset(data_path=args.data_path ,mode='train')
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,pin_memory=True,num_workers=16)
    
    val_data=load_dataset(data_path=args.data_path ,mode='test')
    val_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=16)
    
    #loss function
    criterion = nn.BCEWithLogitsLoss() #適用二分類問題 將model映射成[0,1] 透過sigmoid function
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # put dice scores and losses
    losses=[]
    dice_scores=[]
    
    
    # Define best scores
    best_scores=1
    for epoch in range(args.epochs):
        train_loss=0
        model.train()
        for  sample in tqdm(train_loader): # (資料集個數) *0.9 / batch_size
            #transfer to GPU
            image, mask = sample["image"].to(device), sample["mask"].to(device)
            
            
            pred = model(image.to(dtype=torch.float32)) #already flatten
            
            loss = criterion(pred,mask)
            
            train_loss+=loss.item()
            
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            
             # Compute the gradients for parameters.
            loss.backward()

            # Update the parameters with computed gradients.
            optimizer.step()
        
        train_loss = train_loss / len(train_loader)
        
        model.eval()
        result=0
        
        ###### eval here ############################
        with torch.no_grad():
            for sample in val_loader:
                image, mask = sample["image"].to(device), sample["mask"].to(device)
                pred = model(image.to(dtype=torch.float32))
                pred=torch.sigmoid(pred)
                ######## calculate dice score
                result += dice_score(pred,mask) 
                
            score= result / len(val_loader)
            
        dice_scores.append(score)
        losses.append(train_loss)
        print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss:.5f},dice_score = {score:.2f}")
        if score>best_scores:
            best_scores=score
            torch.save(model.state_dict(), f"{save_path}/{args.model}/model_{args.model}_{score:.2f}.pth")
            print(f"Save the model with score{score:.2f}")
    
    #os.makedirs(save_path, exist_ok=True)    
    
    
    torch.save(model.state_dict(), f"{save_path}/model.pth")
    print(f'model save to {save_path}/{args.model}')
    #turn into numpy file to plot the graph
    np.save(f"{save_path}/{args.model}/losses_witoutDA_{args.model}.npy", losses)
    np.save(f"{save_path}/{args.model}/dice_scores_witoutDA_{args.model}.npy", dice_scores)    
    #assert False, "Not implemented yet!"


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path',  default= '../dataset',type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e',type=int, default=25, help='number of epochs')
    parser.add_argument('--batch_size', '-b',type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', '-lr',type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model',type=str, default="Unet",choices=['Unet', 'ResNet34_Unet'], help='Unet or ResNet34_Unet ')
    


    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    #model = Unet(in_ch=3,out_ch=1)
    if(args.model == "Unet"):
        print("Training Unet")
        model = Unet(in_ch=3,out_ch=1)
    elif(args.model=="ResNet34_Unet"):
        print("Training ResNet34_UNet")
        model=ResNet34_UNet(in_ch=3,out_ch=1)
    train(args,model=model,save_path='../saved_models')