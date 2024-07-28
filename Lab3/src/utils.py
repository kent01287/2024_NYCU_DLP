import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import random

from PIL import Image
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    pred_mask = (pred_mask > 0.5).float() # >0.5 ->1 <=0.5 ->0
    #flatten 
    pred_mask = pred_mask.flatten(start_dim=1)
    gt_mask = gt_mask.flatten(start_dim=1)
    
    #calculate the intersection
    sum_dice_s=0
    for i in range(len(pred_mask)):
        intersection = (pred_mask[i] * gt_mask[i]).sum()
        dice_s = 2*intersection / (pred_mask[i].sum()+gt_mask[i].sum())
        sum_dice_s += dice_s
    return (sum_dice_s / len(pred_mask)).item() # type: ignore

    assert False, "Not implemented yet!"

def show_model_Scroe(np1,np2):
    data1 = np.load(np1)
    data2 = np.load(np2)
    
    
    plt.figure()
    plt.plot(data1, label='Unet')
    

    plt.plot(data2, label='ResNet34_Unet', color='orange')

    
    plt.xlabel("epochs")
    plt.title('Model Score')
    plt.ylabel("scores")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def show_model_loss(np1,np2):
    data1 = np.load(np1)
    data2 = np.load(np2)

    
    plt.figure()

    plt.plot(data1, label='Unet')
    plt.plot(data2, label='ResNet34_Unet', color='orange')
    plt.title('Model loss')
    
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def show_model_DA(np1,np2):
    data1 = np.load(np1)
    data2 = np.load(np2)

    
    plt.figure()

    plt.plot(data1, label='Without DA')
    plt.plot(data2, label='With DA', color='orange')
    plt.title('Model loss')
    
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def show_DA (img):
    # Load the image
    img = Image.open(img)

    # Rotate the image by a specified angle
    rotated = img.rotate(180)

    # Save the rotated image
    #rotated.save('../img/rotated_image.jpg')
    ##########crop the image################
    option=[transforms.CenterCrop(size=(200,200))]
    transform2=option[0]
    if transform2 is not None:
        img=transform2(img)
    
    img.save('../img/cropped.jpg')
        
if __name__ == "__main__":
    '''
    pred_mask = torch.tensor([[[0.9, 0.1, 0.8], [0.4, 0.6, 0.7]], [[0.2, 0.3, 0.1], [0.9, 0.8, 0.9]]], dtype=torch.float32)
    gt_mask = torch.tensor([[[1, 0, 1], [0, 1, 1]], [[0, 0, 0], [1, 1, 1]]], dtype=torch.float32)
    
    dice = dice_score(pred_mask, gt_mask)
    print(f"Dice Score: {dice}")
    print(type(dice))
    '''
    
    # plot the graph
    show_model_loss('../saved_models/Unet/losses_Unet.npy','../saved_models/ResNet34_Unet/losses_ResNet34_Unet.npy')
    show_model_Scroe('../saved_models/Unet/dice_scores_Unet.npy','../saved_models/ResNet34_Unet/dice_scores_ResNet34_Unet.npy')
    
    # show the grpah
    #show_DA('../img/original_image.png')