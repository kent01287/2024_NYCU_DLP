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
        batch_size, num_tokens, dim = x.shape
        
        # Linear transformation
        q = self.query(x)  # (B, T, D)
        k = self.key(x)    # (B, T, D)
        v = self.value(x)  # (B, T, D)
        
        print("After linear transformations:")
        print(f"q shape: {q.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")
        
        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, num_tokens, self.num_heads, dim // self.num_heads).transpose(1, 2)  # (B, nh, T, dim_per_head)
        k = k.view(batch_size, num_tokens, self.num_heads, dim // self.num_heads).transpose(1, 2)  # (B, nh, T, dim_per_head)
        v = v.view(batch_size, num_tokens, self.num_heads, dim // self.num_heads).transpose(1, 2)  # (B, nh, T, dim_per_head)
        
        print("After reshaping and transposing for multi-head attention:")
        print(f"q shape: {q.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")
        
        # Compute attention matrix
        att = q @ k.transpose(2, 3)  # (B, nh, T, T)
        att = att / np.sqrt(k.size(-1))  # Scaling
        
        print("After computing attention matrix and scaling:")
        print(f"att shape: {att.shape}")
        
        att = nn.functional.softmax(torch.tensor(att), dim=-1)  # Convert to tensor and apply softmax
        att = self.dropout(att)
        
        print("After applying softmax and dropout:")
        print(f"att shape: {att.shape}")
        
        y = att @ v  # (B, nh, T, dim_per_head)
        
        print("After computing output of attention:")
        print(f"y shape: {y.shape}")
        
        y = y.transpose(1, 2).contiguous().view(batch_size, num_tokens, dim)  # Re-assemble all head outputs side by side
        print("After reassembling head outputs and projecting:")
        print(f"y shape: {y.shape}")
        
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
    