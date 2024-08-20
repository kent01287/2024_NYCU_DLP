import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import matplotlib.pyplot as plt

#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs']) 

        self.num_image_tokens = configs['num_image_tokens'] #256
        self.mask_token_id = configs['num_codebook_vectors'] #
        self.choice_temperature = configs['choice_temperature'] #4.5
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod #不需創物件即可使用
    def load_vqgan(configs):#模型配置路線與路徑的字典
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        zq, z_indices, _ = self.vqgan.encode(x) #_ is q_loss
        #raise Exception('TODO2 step1-1!')
        return zq,z_indices
    
##TODO2 step1-2:    decide mask schedule
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).
        一開始很高，後面越來越低
        """
        if mode == "linear":
            mask_rate=lambda ratio: 1 - ratio
            #raise Exception('TODO2 step1-2!')
            #return None
        elif mode == "cosine":
            mask_rate=lambda ratio:math.cos(math.pi * ratio / 2)
            #raise Exception('TODO2 step1-2!')
            #return None
        elif mode == "square":
            mask_rate=lambda ratio : 1 - ratio ** 2
            #raise Exception('TODO2 step1-2!')
            #return None
        else:
            raise NotImplementedError
        return mask_rate
        
##TODO2 step1-3:            
    def forward(self, x,ratio): #X (B,C,H,W) 
        '''
         z_indices:(B,num_tokens)
         logits : (B,num_tokens,num_codebook_vector+1)
         x :圖
         ratio : 隨機sample的圖
        '''
        # encode 得codebook indice
        _,z_indices=self.encode_to_z(x) #ground truth  [B*16*16]
        z_indices=z_indices.view(-1,self.num_image_tokens) #[B,256]
        # 隨機sample 一個mask出來
        mask = torch.bernoulli(torch.ones_like(z_indices) * ratio) #1代表被mask 0代表沒有
        mask_bool = mask.to(torch.bool)
        #套上mask
        input = mask_bool * self.mask_token_id + (~mask_bool) *  z_indices 
        #丟給transformer預測
        logits = self.transformer(input)    # transformer predict the probability of tokens
        
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self,z_indices,ratio,mask,mask_num): #上一次的預測tokens
        '''
        z_indices: [B,256]
        ratio: 控制要被mask的比例
        mask: 對應的遮罩 
        1.根據mask將z_indices套上mask 並丟進去transformer得到結果 
        2.將結果softmax並取得每個token_value最大值 即可得到最大機率的index
        3. add back the original token values that were not masked to the predicted tokens
        '''
        # 取得被mask的地方 
        ################################### 處理 inpainting的原圖 ###########################################
        #套上mask
        input=mask*self.mask_token_id+(~mask)*z_indices
        #丟給Transformer
        logits = self.transformer(input)#[B,num_tokens,num_codeBlockVector+1]
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.nn.functional.softmax(logits, dim=-1) 

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1) #value index (B,256),(B,256) #最大值即codebook對應的label
        
        ################################# 處理被 mask圖 修復進度 需要input到下一個 mask_b ###########################################
        #predicted probabilities add temperature annealing gumbel noise as confidence
        #Gumbel 噪聲來增加隨機性
        #溫度參數控制隨機性，低溫度使預測更確定，高溫度增加隨機性
        g = torch.distributions.Gumbel(0, 1).sample(sample_shape=z_indices_predict_prob.shape).to(z_indices_predict_prob.device)  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        
        #若mask是0 則取無限大，只想關注mask=1的部分
        confidence = torch.where(mask == 0, torch.tensor(float('inf')).to(confidence.device), confidence) #(1,256)
        #sort the confidece for rank
        n = math.ceil(mask_num * self.gamma(ratio)) #算有幾個1
        values,indices=torch.topk(confidence,n,largest=False) #由小排到大將mask=1 predict的結果
        max_values=values[0,-1] #取出這些最小值的最大值
        mask_bc=confidence<=max_values #若比<=max_values 設true 代表要繼續predice , > 則設成false 不用predict
        
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        z_indices_predict = mask * (z_indices_predict) + (~mask) * z_indices #把1的改成z_indices_preict
        return z_indices_predict, mask_bc
    def plot_gamma(self, total_iter, save_path):
        ratio_list = []
        cosine_list, linear_list, square_list = [], [], []
        
        cosine_func = self.gamma_func("cosine")
        linear_func = self.gamma_func("linear")
        square_func = self.gamma_func("square")
        
        for i in range(total_iter):
            ratio = i / total_iter
            ratio_list.append(ratio)
            cosine_ratio = cosine_func(ratio)
            cosine_list.append(cosine_ratio)
            linear_ratio = linear_func(ratio)
            linear_list.append(linear_ratio)
            square_ratio = square_func(ratio)
            square_list.append(square_ratio)
        
        # Plot using ratio_list as x-axis data
        plt.plot(ratio_list, cosine_list, marker='o', label='cosine')
        plt.plot(ratio_list, linear_list, marker='o', label='linear')
        plt.plot(ratio_list, square_list, marker='o', label='square')
        
        plt.title("Mask scheduling curve")
        plt.xlabel("t/T")
        plt.ylabel("Gamma function")
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='jpg')  # Save as JPG with high quality
            
    '''
    mask_bc代表目前decode的黑白圖 mask_bc變成下一個iteration的mask圖，用於修復
    z_indices_predict output result
    '''
        
    
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    
if __name__=='__main__':
    img = torch.randn(10, 3, 64, 64)
    cfg=yaml.safe_load(open('./config/MaskGit.yml', 'r'))
    model=MaskGit(cfg['model_param'])
    #ratio=np.random()
    #logits, z_indices=model(img,ratio)
    #print(logits.shape) #(10,256,1025)
    #print(z_indices.shape)#(10,256)
    model.plot_gamma(total_iter=10,save_path="./img/ratio.jpg")
        
