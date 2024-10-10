# Modified from:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vivit.py

import torch
from torch import nn
from einops import rearrange

# helpers
def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class Attention(nn.Module): # with pre norm
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ViViT(nn.Module):
    def __init__(
        self,
        *,
        frames = 16,
        frame_patch_size = 1,
        dim = 768,
        heads = 12,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        num_query_tokens = 32,
        out_dim = 4096,
        no_pos = False,
        no_spatial = False,
        no_temporal = False,
        no_projector = False,
    ):
        super().__init__()
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        num_frame_patches = (frames // frame_patch_size)

        self.no_pos = no_pos
        if not self.no_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_query_tokens, dim))
            self.dropout = nn.Dropout(emb_dropout)

        self.no_spatial = no_spatial
        if not self.no_spatial:
            self.spatial_attention = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout)

        self.no_temporal = no_temporal
        if not self.no_temporal:
            self.temporal_attention = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout)
        
        self.no_projector = no_projector
        if not self.no_projector:
            self.proj = nn.Linear(dim, out_dim)
        
    def forward(self, x):
        # print("Before adding pos_embedding")
        # print(x.shape)
        
        if not (self.no_pos and self.no_temporal and self.no_temporal):
            f = 16
            x = rearrange(x, '(b f) n d -> b f n d', f = f)
            b, f, n, d = x.shape

        if not self.no_pos:
            x = x + self.pos_embedding[:, :f, :n] # to handle f < 16
            # print(f"self.pos_embedding.shape = {self.pos_embedding.shape}")
            # print(f"self.pos_embedding[:, :f, :n].shape = {self.pos_embedding[:, :f, :n].shape}")
            # # print(f"self.pos_embedding[:, :f, :n] = {self.pos_embedding[:, :f, :n]}")    
            x = self.dropout(x)
        
        if not self.no_spatial: 
            # spatial attention
            x = rearrange(x, 'b f n d -> (b f) n d')
            # print("\nBefore spatial_attention, (b f) n d")
            # print(x.shape)
            x = self.spatial_attention(x) + x # residual connection
            # print("After spatial_attention")
            # print(x.shape)

            x = rearrange(x, '(b f) n d -> b f n d', b = b)
            # print("\nRecover, b f n d")
            # print(x.shape)

        if not self.no_temporal:
            # temporal attention
            x = rearrange(x, 'b f n d -> (b n) f d')
            # print("\nBefore temporal_attention, (b n) f d")
            # print(x.shape)
            x = self.temporal_attention(x) + x # residual connection
            # print("After temporal_attention")
            # print(x.shape)
            
            x = rearrange(x, '(b n) f d -> b f n d', b = b)
            # print("\nRecover, b f n d")
            # print(x.shape)

        if not self.no_projector:
            x = self.proj(x)
            # print("After proj")
            # print(x.shape)
        
        if not (self.no_pos and self.no_temporal and self.no_temporal):
            x = rearrange(x, 'b f n d -> (b f) n d')
        
        return x
        
if __name__ == "__main__":
    model_params = {
        "frames": 16,
        "frame_patch_size": 1,
        "dim": 768,
        # "spatial_depth": 1,
        # "temporal_depth": 1,
        "heads": 12,
        "dim_head": 64,
        "dropout": 0.,
        "emb_dropout": 0.,
        "num_query_tokens": 32,
        "out_dim": 4096
    }

    # Create the ViViT model
    vivit = ViViT(**model_params)

    # input: QFormer output (num_query_tokens = 32, hidden_size = 768)
    # Generate a random input (b, f, n, d)
    # 2, 8, 32, 768
    input = torch.randn(2, 8, 32, 768) 

    # # print the model output
    # print('ViViT OUTPUT:')
    output = vivit(input)
    
# ViViT OUTPUT:
# Before adding pos_embedding
# torch.Size([2, 8, 32, 768])
# self.pos_embedding.shape = torch.Size([1, 8, 32, 768])
# self.pos_embedding[:, :f, :n].shape = torch.Size([1, 8, 32, 768])

# Before spatial_attention, (b f) n d
# torch.Size([16, 32, 768])
# After spatial_attention
# torch.Size([16, 32, 768])

# Recover, b f n d
# torch.Size([2, 8, 32, 768])

# Before temporal_attention, (b n) f d
# torch.Size([64, 8, 768])
# After temporal_attention
# torch.Size([64, 8, 768])

# Recover, b f n d
# torch.Size([2, 8, 32, 768])
# After proj
# torch.Size([2, 8, 32, 4096])
