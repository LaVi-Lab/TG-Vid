from rotary_embedding_torch import RotaryEmbedding
import torch
import torch.nn as nn

# Modify from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# Modify from: https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/layers/blocks.py
class MultiheadAttentionWithRope(nn.Module):
    def __init__(
        self,
        dim: int = 1408,
        num_heads: int = 16,
        bias: bool = True,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = False,
        dropout: float = 0.0,
        use_rope: bool = False,
        use_RMSNorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=use_qkv_bias)
        
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            if not use_RMSNorm:
                self.q_norm = nn.LayerNorm(self.head_dim)
                self.k_norm = nn.LayerNorm(self.head_dim)
            else:
                self.q_norm = RMSNorm(self.head_dim)
                self.k_norm = RMSNorm(self.head_dim)
            
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
            # self.rotary_emb = self.rotary_emb.rotate_queries_or_keys

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # WARNING: this may be a bug
        if self.use_rope:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        dtype = q.dtype
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        attn = self.dropout(attn)
        x = attn @ v

        x_output_shape = (B, N, C)
        x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    hidden_size = 1408
    num_heads = 16
    
    # spatial
    spatial_attn = MultiheadAttentionWithRope(
        hidden_size,
        num_heads=num_heads,
        use_qkv_bias=True,
        use_qk_norm=True,
        use_rope=False,
    )
    
    # temporal
    temporal_attn = MultiheadAttentionWithRope(
        hidden_size,
        num_heads=num_heads,
        use_qkv_bias=True,
        use_qk_norm=True,
        use_rope=True,
    )

    # Test spatial attention
    input_tensor = torch.randn(2, 10, hidden_size)  # Random input tensor
    spatial_output = spatial_attn(input_tensor)
    print("Spatial Attention Output Shape:", spatial_output.shape)

    # Test temporal attention
    input_tensor = torch.randn(2, 10, hidden_size)  # Random input tensor
    temporal_output = temporal_attn(input_tensor)
    print("Temporal Attention Output Shape:", temporal_output.shape)

# Spatial Attention Output Shape: torch.Size([2, 10, 1408])
# Temporal Attention Output Shape: torch.Size([2, 10, 1408])