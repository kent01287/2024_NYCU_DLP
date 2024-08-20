import torch
import torch.nn as nn
import yaml
from .modules.transform import Encoder, Decoder, Codebook

__all__ = [
    "VQGAN"
]
#quantization loss 輸入特徵圖 與 量化後的特徵圖差異
class VQGAN(nn.Module):
    def __init__(self, configs):
        super(VQGAN, self).__init__()
        
        dim = configs['latent_dim'] #256
        self.encoder = Encoder(configs) #[128, 128, 128, 256, 256, 512]
        self.decoder = Decoder(configs) #[128, 128, 256, 256, 512]
        self.codebook = Codebook(configs) #1024
        self.quant_conv = nn.Conv2d(dim, dim, 1) #接在encoder後
        self.post_quant_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, imgs):
        # 編碼影像
        encoded_images = self.encoder(imgs)
        print(f"Encoded images size: {encoded_images.size()}")  # [10,256,64,64]

        # 量化編碼影像
        quantized_encoded_images = self.quant_conv(encoded_images)
        print(f"Quantized encoded images size: {quantized_encoded_images.size()}")  # [10,256,64,64]

        # 從碼本找到最近的映射、索引及量化損失
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        print(f"Codebook mapping size: {codebook_mapping.size()}")  # [10,256,64,64]
        print(f"Codebook indices size: {codebook_indices.size()}")  # [40960]
        print(f"Quantization loss: {q_loss}")  # 印出量化損失

        # 量化後特徵圖進行卷積
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping)
        print(f"Quantized codebook mapping size: {quantized_codebook_mapping.size()}")  #[10, 256, 64, 64]

        # 還原影像
        decoded_images = self.decoder(quantized_codebook_mapping) #[10, 3 64, 64]
        print(f"Decoded images size: {decoded_images.size()}")  # 印出還原後影像的尺寸

        return decoded_images, codebook_indices, q_loss
    def encode(self, x):
        # 編碼影像
        encoded_images = self.encoder(x)
        #print(f"Encoded images size: {encoded_images.size()}")  # 印出編碼後影像的尺寸，預期是 [B, 256, 16, 16]

        # 量化編碼影像
        quantized_encoded_images = self.quant_conv(encoded_images)
        #print(f"Quantized encoded images size: {quantized_encoded_images.size()}")  # 印出量化編碼後影像的尺寸，預期是 [B, 256, 16, 16]

        # 映射到最近的 codebook
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
       #print(f"Codebook mapping size: {codebook_mapping.size()}")  # 印出碼本映射的尺寸，預期是 [B, 256, 16, 16]
        #print(f"Codebook indices size: {codebook_indices.size()}")  # 印出碼本索引的尺寸，預期是 [2560] B*16*16
        #print(f"Quantization loss: {q_loss}")  # 印出量化損失

        return codebook_mapping, codebook_indices, q_loss
    def decode(self, z):
        quantized_codebook_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_codebook_mapping)
        return decoded_images
    #平衡 NLL loss and Gernerator loss 動態來調整
    def calculate_lambda(self, nll_loss, g_loss):
        last_layer = self.decoder.model[-1] #decoder網路最後一層
        last_layer_weight = last_layer.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0] #計算nll loss 相對於最後一層的 gradient
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]#計算Gernerator loss 相對於最後一層的 gradient

        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4) 
        λ = torch.clamp(λ, 0, 1e4).detach()#限制在0~10000 並使用detach分離出來
        #discriminator weight=0.8
        return 0.8 * λ #調整損失函數的權重

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path), strict=True)
        print("Loaded Checkpoint for VQGAN....")

if __name__=='__main__':
    img = torch.randn(10, 3, 64, 64)
    cfg=yaml.safe_load(open('./models/VQGAN/config/VQGAN.yml', 'r'))
    model=VQGAN(cfg['model_param'])
    '''
    decoded_images, codebook_indices, q_loss=model(img)
    print(decoded_images.shape) #[10,3,256,256]
    print(codebook_indices.shape)#[40960]
    '''
    ###############################
    codebook_mapping, codebook_indices, q_loss=model.encode(img)