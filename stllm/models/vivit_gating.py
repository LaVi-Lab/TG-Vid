import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

class GatingLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, residual, x):
        # gating: x = residual + gate(residual, x)*x
        gating_weights = torch.sigmoid(self.linear(torch.cat([residual, x], dim=-1)))
        # print("gating_weights.shape, x.shape = ", gating_weights.shape, x.shape)
        x = gating_weights * x  # element-wise product
        x = residual + x
        return x

# Modify from: https://github.com/baaivision/EVA/blob/master/EVA-02/asuka/modeling_finetune.py
class MLPSwiGLU(nn.Module):
    def __init__(self, hidden_dim=1408, mlp_ratio=4.0, dropout=0., use_subln=False):
        super().__init__()
        
        self.w1 = nn.Linear(hidden_dim, int(mlp_ratio * hidden_dim))
        self.w2 = nn.Linear(hidden_dim, int(mlp_ratio * hidden_dim))
        self.activation = nn.SiLU()
        
        self.use_subln = use_subln
        if self.use_subln:
            self.ffn_ln = nn.LayerNorm(int(mlp_ratio * hidden_dim))
            
        self.w3 = nn.Linear(int(mlp_ratio * hidden_dim), hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.activation(x1) * x2
        
        if self.use_subln:
            x = self.ffn_ln(hidden)
        else:
            x = hidden
            
        x = self.w3(x)
        x = self.dropout(x)
        return x


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
# Self-Attention
class MultiheadSelfAttentionWithRope(nn.Module):
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
            self.q_norm = nn.LayerNorm(self.head_dim) if not use_RMSNorm else RMSNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim) if not use_RMSNorm else RMSNorm(self.head_dim)
            
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


# Spatial-Temporal Gating
class STGLayer(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0., 
                 is_gating=False, 
                 is_temporal_first=False,
                 use_mlp_swiglu=False,
                 use_mlp_swiglu_subln=False,
                 use_RMSNorm=False,
                 use_temporal_attn_rope=False,
                 use_spatial_attn_rope=False,
                 use_rope_qkv_bias=False,
                 use_rope_qkv_norm=False,
                 no_mlp=False,
                 no_spatial=False,
                 no_temporal=False,
                no_spatial_gating=False,
                no_temporal_gating=False,
                no_mlp_gating=False,
                 ):
        super().__init__()

        self.is_gating = is_gating
        self.is_temporal_first = is_temporal_first
        self.use_mlp_swiglu = use_mlp_swiglu
        self.use_temporal_attn_rope = use_temporal_attn_rope
        self.use_spatial_attn_rope = use_spatial_attn_rope
        self.no_mlp = no_mlp
        self.no_spatial = no_spatial
        self.no_temporal = no_temporal
        self.no_spatial_gating=no_spatial_gating
        self.no_temporal_gating=no_temporal_gating
        self.no_mlp_gating=no_mlp_gating
        
        if not self.is_temporal_first: # STG
            if not self.no_spatial:
                self.pre_norm1 = nn.LayerNorm(hidden_dim) if not use_RMSNorm else RMSNorm(hidden_dim)
                if not self.use_spatial_attn_rope:
                    self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                else:
                    self.spatial_attn = MultiheadSelfAttentionWithRope(dim=hidden_dim, num_heads=num_heads, dropout=dropout, 
                                                                    use_qkv_bias=use_rope_qkv_bias, 
                                                                    use_qk_norm=use_rope_qkv_norm, 
                                                                    use_RMSNorm=use_RMSNorm
                                                                    )
                if self.is_gating and not self.no_spatial_gating: 
                    self.gating1 = GatingLayer(hidden_dim)

            if not self.no_temporal:
                self.pre_norm2 = nn.LayerNorm(hidden_dim) if not use_RMSNorm else RMSNorm(hidden_dim)
                if not self.use_temporal_attn_rope:
                    self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                else:
                    self.temporal_attn = MultiheadSelfAttentionWithRope(dim=hidden_dim, num_heads=num_heads, dropout=dropout, 
                                                                    use_qkv_bias=use_rope_qkv_bias, 
                                                                    use_qk_norm=use_rope_qkv_norm, 
                                                                    use_RMSNorm=use_RMSNorm
                                                                    )

                if self.is_gating and not self.no_temporal_gating:
                    self.gating2 = GatingLayer(hidden_dim)
                    
        else: # TSG
            if not self.no_temporal:
                self.pre_norm1 = nn.LayerNorm(hidden_dim) if not use_RMSNorm else RMSNorm(hidden_dim)
                if not self.use_temporal_attn_rope:
                    self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                else:
                    self.temporal_attn = MultiheadSelfAttentionWithRope(dim=hidden_dim, num_heads=num_heads, dropout=dropout, 
                                                                    use_qkv_bias=use_rope_qkv_bias, 
                                                                    use_qk_norm=use_rope_qkv_norm, 
                                                                    use_RMSNorm=use_RMSNorm
                                                                    )
                if self.is_gating and not self.no_temporal_gating: 
                    self.gating1 = GatingLayer(hidden_dim)

            if not self.no_spatial:
                self.pre_norm2 = nn.LayerNorm(hidden_dim) if not use_RMSNorm else RMSNorm(hidden_dim)
                
                if not self.use_spatial_attn_rope:
                    self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                else:
                    self.spatial_attn = MultiheadSelfAttentionWithRope(dim=hidden_dim, num_heads=num_heads, dropout=dropout, 
                                                                    use_qkv_bias=use_rope_qkv_bias, 
                                                                    use_qk_norm=use_rope_qkv_norm, 
                                                                    use_RMSNorm=use_RMSNorm
                                                                    )
                
                if self.is_gating and not self.no_spatial_gating: 
                    self.gating2 = GatingLayer(hidden_dim)
        
        if not self.no_mlp:
            self.pre_norm3 = nn.LayerNorm(hidden_dim) if not use_RMSNorm else RMSNorm(hidden_dim)
            if self.use_mlp_swiglu:
                self.mlp = MLPSwiGLU(hidden_dim=hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout, use_subln=use_mlp_swiglu_subln)
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_dim, int(mlp_ratio * hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(mlp_ratio * hidden_dim), hidden_dim),
                    nn.Dropout(dropout)
                )
            if self.is_gating and not self.no_mlp_gating:
                self.gating3 = GatingLayer(hidden_dim)

    def forward(self, x):
        # Get shape, (batch, num_frames, num_patch_tokens, hidden_dim)
        B, T, L, D = x.shape
                    
        if not self.is_temporal_first:
            # Spatial attention
            x = rearrange(x, 'B T L D -> (B T) L D')
            residual = x
            if not self.no_spatial:
                if not self.use_spatial_attn_rope:
                    x, _ = self.spatial_attn(x, x, x)
                else:
                    x = self.spatial_attn(x)
                
                if self.is_gating and not self.no_spatial_gating:
                    x = self.gating1(residual, x)
                else:
                    x = residual + x
                    
            # Temporal attention
            x = rearrange(x, '(B T) L D -> (B L) T D', T=T)
            residual = x
            if not self.no_temporal:
                if not self.use_temporal_attn_rope:
                    x, _ = self.temporal_attn(x, x, x)
                else:
                    x = self.temporal_attn(x)
                if self.is_gating and not self.no_temporal_gating:
                    x = self.gating2(residual, x)
                else:
                    x = residual + x
        else:
            # Temporal attention
            x = rearrange(x, 'B T L D -> (B L) T D')
            residual = x
            if not self.no_temporal:
                if not self.use_temporal_attn_rope:
                    x, _ = self.temporal_attn(x, x, x)
                else:
                    x = self.temporal_attn(x)
                if self.is_gating and not self.no_temporal_gating:
                    x = self.gating2(residual, x)
                else:
                    x = residual + x
            
            # Spatial attention
            x = rearrange(x, '(B L) T D -> (B T) L D', L=L)
            residual = x
            if not self.no_spatial:
                if not self.use_spatial_attn_rope:
                    x, _ = self.spatial_attn(x, x, x)
                else:
                    x = self.spatial_attn(x)
                
                if self.is_gating and not self.no_spatial_gating:
                    x = self.gating1(residual, x)
                else:
                    x = residual + x
            
        # MLP
        if not self.is_temporal_first:
            x = rearrange(x, '(B L) T D -> B T L D', L=L)
        else:
            x = rearrange(x, '(B T) L D -> B T L D', T=T)
        
        if not self.no_mlp:
            x = self.pre_norm3(x)
            residual = x
            x = self.mlp(x)
            if self.is_gating and not self.no_mlp_gating:
                x = self.gating3(residual, x)
            else:
                x = residual + x

        return x

# ST-Gating
class STG(nn.Module):
    def __init__(self, num_layers=1, hidden_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0., 
                 is_gating=False, 
                 is_temporal_first=False, 
                 no_spatial=False, 
                 no_temporal=False,
                 no_mlp=False, 
                 use_pos_patch_level=False, 
                 use_pos_frame_level=False, 
                 use_pos_absolute=False,
                 num_frames=16, num_patch_tokens=256+1,
                 use_xavier_init=False, 
                 use_mlp_swiglu=False,
                 use_mlp_swiglu_subln=False,
                 use_RMSNorm=False,
                 use_temporal_attn_rope=False,
                 use_spatial_attn_rope=False,
                 use_rope_qkv_bias=False,
                 use_rope_qkv_norm=False,
                 no_spatial_gating=False,
                no_temporal_gating=False,
                no_mlp_gating=False,
):
        super().__init__()
        
        self.use_pos_patch_level = use_pos_patch_level
        self.use_pos_frame_level = use_pos_frame_level
        self.use_pos_absolute = use_pos_absolute
        
        if self.use_pos_patch_level:
            # input.shape = (batch, num_frames, num_patch_tokens, hidden_dim)
            # num_patch_tokens = 256 + 1, including cls token
            # each patch token has a position embedding
            if self.use_pos_absolute:
                # absolute, sinusoidal position encoding
                self.pos_embedding = self._get_sinusoidal_position_encoding(num_frames*num_patch_tokens, hidden_dim)
                self.pos_embedding = rearrange(self.pos_embedding, '(T L) D -> T L D', L=num_patch_tokens)
                self.pos_embedding = self.pos_embedding.unsqueeze(0)
            else:
                self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patch_tokens, hidden_dim))
        elif self.use_pos_frame_level: 
            # frame-level, temporal position embedding
            # each frame has a position embedding
            # all patch tokens in the same frame share the same position embedding
            if self.use_pos_absolute:
                # absolute, sinusoidal position encoding
                self.pos_embedding = self._get_sinusoidal_position_encoding(num_frames, hidden_dim)
                self.pos_embedding = self.pos_embedding.unsqueeze(0).unsqueeze(2)
            else:
                self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, 1, hidden_dim))
                    
        # if no_spatial_gating or no_temporal_gating or no_mlp_gating:
        #     raise ValueError("STG not implement no_spatial_gating or no_temporal_gating or no_mlp_gating")
        
        self.layers = nn.ModuleList([
                STGLayer(hidden_dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, 
                            is_gating=is_gating, is_temporal_first=is_temporal_first, 
                            use_mlp_swiglu=use_mlp_swiglu, use_mlp_swiglu_subln=use_mlp_swiglu_subln,
                            use_RMSNorm=use_RMSNorm,
                            use_temporal_attn_rope=use_temporal_attn_rope, 
                            use_spatial_attn_rope=use_spatial_attn_rope,
                            use_rope_qkv_bias=use_rope_qkv_bias,
                            use_rope_qkv_norm=use_rope_qkv_norm,
                            no_mlp=no_mlp,
                            no_spatial=no_spatial,
                            no_temporal=no_temporal,
                            no_spatial_gating=no_spatial_gating,
                            no_temporal_gating=no_temporal_gating,
                            no_mlp_gating=no_mlp_gating,
                            ) 
                for _ in range(num_layers)
            ])

        self.use_xavier_init = use_xavier_init
        if self.use_xavier_init:
            self.apply(self._xavier_normal_init)

    def _xavier_normal_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.0)

    def _get_sinusoidal_position_encoding(self, max_len, dim):
        # Transformer, Absolute Position Encoding
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        if dim % 2 != 0: 
            raise ValueError(f"Cannot create sinusoidal position encoding for odd dimension: {dim}")
        
        with torch.no_grad():
            position_encoding = torch.zeros(max_len, dim) # max_len, dimension
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape = (max_len, 1)

            _2i = torch.arange(0, dim, 2, dtype=torch.float)
            position_encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / dim)))
            position_encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / dim)))

        return position_encoding

    def forward(self, x):
        # Get shape, (batch, num_frames, num_patch_tokens, hidden_dim)
        B, T, L, D = x.shape
                
        if self.use_pos_patch_level or self.use_pos_frame_level:
            if self.use_pos_patch_level:
                assert x.shape[2] == self.pos_embedding.shape[2], f"Number of patch tokens should be equal to the number of position embeddings. {x.shape[2]} != {self.pos_embedding.shape[2]}"
            # (1, num_frames, num_patch_tokens, hidden_dim) or (1, num_frames, 1, hidden_dim)
            # x = x + self.pos_embedding 
            if self.use_pos_absolute:
                self.pos_embedding = self.pos_embedding.to(x.device)
            x = x + self.pos_embedding[:, :T, :L] # to handle T < 16

        # STG layers
        for layer in self.layers:
            x = layer(x)

        return x