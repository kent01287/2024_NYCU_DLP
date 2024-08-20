import torch.nn as nn
import torch
import math
import numpy as np
#TODO1
#用q去找相關的k
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        # For linear transformation
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(attn_drop)
        # Output projection
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim) '''
        '''
        1.Linear transformation for qkv vector
        2.qkv dim transform to (B,n_h,n_t,d_p_h)
        3.calculate the attention matrix
        4.apply softmax (歸一化)
        5.return to origin shape and do projection
        '''
        batch_size, num_tokens, dim = x.shape
        
        # Linear transformation
        q = self.query(x)  # q shape: (batch_size, num_tokens, dim)
        k = self.key(x)    # k shape: (batch_size, num_tokens, dim)
        v = self.value(x)  # v shape: (batch_size, num_tokens, dim)
        # Reshape and transpose for multi-head attention
        # Reshape q, k, v to split dim into multiple heads and then transpose to get shape
        # (batch_size, num_heads, num_tokens, dim_per_head)
        q = q.view(batch_size, num_tokens, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(batch_size, num_tokens, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(batch_size, num_tokens, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        
        # Compute attention matrix
        # Multiply q with transposed k and then scale by the square root of the key dimension size
        att = q @ k.permute(0, 1, 3, 2)  # att shape: (batch_size, num_heads, num_tokens, num_tokens)
        #防止attention值太大
        att = att / np.sqrt(k.size(-1))  # Scaling to prevent extremely large values in softmax
        
        # Convert att to a tensor and apply softmax along the last dimension to get attention weights
        att = nn.functional.softmax(torch.tensor(att), dim=-1)
        att = self.dropout(att)
        
        # Multiply the attention weights with the value vectors to get the attention output
        y = att @ v  # y shape: (batch_size, num_heads, num_tokens, dim_per_head)
        
        # Transpose y back to its original shape and combine the head outputs into a single dimension
        y = y.transpose(1, 2).contiguous().view(batch_size, num_tokens, dim) #確保y是連續的
        
        # Project the combined output to the original dimension using a linear layer
        return self.proj(y)
class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)

if __name__ == '__main__':
    # parameter setting
    batch_size = 2
    num_tokens = 10
    dim = 768
    num_heads = 16

    # create data
    x = torch.randn(batch_size, num_tokens, dim)  # size(B,T,D)

    
    mha = MultiHeadAttention(dim=dim, num_heads=num_heads)

    # 
    output = mha(x)

    print("Final output shape:")
    print(output.shape)  # 應是 (B, T, D)
    