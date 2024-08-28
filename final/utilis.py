import os
from PIL import Image
import torch
import torchvision.transforms as transforms

def add_noise(x, noise_level, noise_type):
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level/255, x.shape) # mu=0
        noisy = torch.clamp(noisy, 0, 1) # 限制 0~1
    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x) / noise_level
    return noisy


image_folder_path = "./Mcmaster18_Dataset/"   
output_folder_path = "./Mcmaster18_Dataset/val_noisy"  


os.makedirs(output_folder_path, exist_ok=True)


transform = transforms.ToTensor()

transform_to_pil = transforms.ToPILImage()

i=0
for image_file in os.listdir(image_folder_path):
    if image_file.endswith(".tif"):
        image_path = os.path.join(image_folder_path, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)

        
        noisy_image_tensor = add_noise(image_tensor, noise_level=25, noise_type='gauss')
    
        noisy_image = transform_to_pil(noisy_image_tensor)
        noisy_image.save(os.path.join(output_folder_path,f"val_noisy_{i}.png"))
        
        i+=1

print("All images have been processed and saved.")