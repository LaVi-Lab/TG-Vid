import logging
import random
import re
import os
import math
import einops
import ast

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss

from stllm.common.registry import registry
from stllm.models.utils import RandomMaskingGenerator, get_sinusoid_encoding_table
from stllm.models.blip2 import Blip2Base, disabled_train
from stllm.models.peft_model import replace_peftmodel_with_sample_input
from stllm.models.base_model import BaseModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaModel
#from stllm.models.modeling_llama_mem import LlamaForCausalLM, LlamaModel
#from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import LlamaTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
)

from stllm.models.vivit_qformer import Attention, ViViT
from stllm.models.vivit_gating import STG

class StllmConfig(LlamaConfig):
    model_type = "st_llm_hf"


class Linear_Decoder(nn.Module):
    def __init__(self, output_dim=4096, embed_dim=4096):
        super().__init__()
        self.head = nn.Linear(embed_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.norm(self.head(x))
        return x

class STLLMLlamaModel(LlamaModel):
    config_class = StllmConfig
    def __init__(self, config: LlamaConfig):  # TODO: Remove unused params
        super(STLLMLlamaModel, self).__init__(config)
    
    def initialize_vision_modules(self, cfg):
        self.stllm_model = STLLMModel.from_config(cfg)
        if cfg.get("qformer_text_input", False):
            self.resize_token_embeddings(len(self.stllm_model.llama_tokenizer))
        self.stllm_model.embed_tokens = self.embed_tokens
    
    def forward(self, samples=None, inputs_embeds=None, **kwargs):
        if samples is None:
            return super(STLLMLlamaModel, self).forward(inputs_embeds=inputs_embeds, **kwargs)
        
        inputs_embeds, attention_mask, unmask_inputs_embeds, unmask_attention_mask, labels = self.stllm_model(samples)
        output_hidden_states = not (unmask_inputs_embeds is None)
        outputs = super(STLLMLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        if unmask_inputs_embeds is None:
            return outputs, None, labels
        
        img_start = 0 if self.stllm_model.qformer_text_input else 8
        mask_output = outputs.hidden_states[-1]
        B, _, D = mask_output.size()
        mask_img_output = mask_output[:,img_start:img_start+self.stllm_model.mask_img_len]
        if hasattr(self.stllm_model, "mvm_decoder"):
            mask_img_output = self.stllm_model.mvm_decoder(mask_img_output)

        with torch.no_grad():
            unmask_outputs = super(STLLMLlamaModel, self).forward(
                inputs_embeds=unmask_inputs_embeds,
                attention_mask=unmask_attention_mask,
                return_dict=True, use_cache=False,
                output_hidden_states=True,
            )
        unmask_output = unmask_outputs.hidden_states[-1]
        unmask_img_output = unmask_output[:,img_start:img_start+self.stllm_model.img_len]
        unmask_img_output = unmask_img_output[~(self.stllm_model.mask.squeeze(1))].reshape(B, -1, D)

        mask_img_output = mask_img_output / mask_img_output.norm(dim=-1, keepdim=True)
        unmask_img_output = unmask_img_output / unmask_img_output.norm(dim=-1, keepdim=True)
        loss_mvm = (2 - 2 * (mask_img_output * unmask_img_output).sum(dim=-1)).mean()
        return outputs, loss_mvm, labels

@registry.register_model("st_llm_hf")
class STLLMForCausalLM(LlamaForCausalLM, BaseModel):
    config_class = StllmConfig
    PRETRAINED_MODEL_CONFIG_DICT = {
        "instructblip_vicuna0": "configs/models/instructblip_vicuna0.yaml",
        "instructblip_vicuna0_btadapter": "configs/models/instructblip_vicuna0_btadapter.yaml",
        "minigpt4_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "minigpt4_vicuna0_btadapter": "configs/models/minigpt4_vicuna0_btadapter.yaml",
    }

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = STLLMLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()  

    def get_model(self):
        return self.model
    
    def forward(self, samples=None, inputs_embeds=None, **kwargs):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if samples is None:
            return super(STLLMForCausalLM, self).forward(inputs_embeds=inputs_embeds, **kwargs)
        outputs, loss_pretrain, labels = self.model(samples)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if loss_pretrain is not None:
            loss += loss_pretrain

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    @classmethod
    def get_state_dict(self, path, prefix='pytorch_model'):
        pattern = re.compile(f'{prefix}-(\d+)-of-(\d+).bin')
        matching_files = [filename for filename in os.listdir(path) if pattern.match(filename)]

        model_state_dict = {}
        for model_path in matching_files:
            partial_state_dict = torch.load(os.path.join(path,model_path), map_location=torch.device('cpu'))
            model_state_dict.update(partial_state_dict)
        return model_state_dict

    @classmethod
    def from_config(cls, cfg):
        llama_model = cfg.get("llama_model")
        
        model = cls.from_pretrained(llama_model, torch_dtype=torch.float16)        
        lora_r = cfg.get("lora_r", 0)
        lora_alpha = cfg.get("lora_alpha", 32)
        if lora_r > 0:
            replace_peftmodel_with_sample_input()
            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, loraconfig)

        model.get_model().initialize_vision_modules(cfg)
        if cfg.get("qformer_text_input", False):
            model.resize_token_embeddings(model.config.vocab_size)
        if cfg.get("freeze_LLM",True):
            for name, param in model.named_parameters():
                if 'stllm_model' not in name and 'lora' not in name:
                    param.requires_grad = False
        if cfg.get("use_grad_checkpoint",True):
            model.gradient_checkpointing_enable()

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("STLLMForCausalLM Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            if os.path.isdir(ckpt_path):
                ckpt = cls.get_state_dict(ckpt_path)
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu")
            if 'model' in ckpt:
                ckpt = ckpt['model']
            if 'llm_proj.weight' in ckpt:
                # print("STLLMForCausalLM 'llm_proj.weight' in ckpt")
                # print("Shape of 'llm_proj.weight':", ckpt['llm_proj.weight'].shape)
                # print("Shape of 'llm_proj.bias':", ckpt['llm_proj.bias'].shape)
                # Shape of 'llm_proj.weight': torch.Size([4096, 768])
                # Shape of 'llm_proj.bias': torch.Size([4096])
                ckpt['llama_proj.weight'] = ckpt.pop('llm_proj.weight')
                ckpt['llama_proj.bias'] = ckpt.pop('llm_proj.bias')
            msg = model.load_state_dict(ckpt, strict=False)
            # for name in ckpt:
            #     if name in model.state_dict():
            #         print(f"Parameter {name} initialized from ckpt.")

        return model

class STLLMModel(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """
    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        pre_encoding=False,
        use_mask=False,
        mvm_decode=False,
        video_input=None,
        residual_size=4,
        qformer_text_input=False,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        has_qformer=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        max_txt_len=32,
        end_sym='\n',
        use_mlp=False,
        use_vivit=False,
        use_vivit_no_pos=False,
        use_vivit_no_spatial=False,
        use_vivit_no_temporal=False,
        use_st_attn_pre = False,
        use_st_attn_post = False,
        use_st_attn_pos = False,
        use_st_attn_no_spatial = False,
        use_st_attn_pre_two_layers = False,
        st_attn_pre_depth = -1,
        st_attn_pre_num_heads = 12,
        st_attn_post_num_heads = 32,
        use_stg_after_qformer = False,
        use_stg_after_vit = False,
        use_stg_after_vitln = False,
        stg_num_layers = -1,
        stg_is_gating = False,
        stg_is_temporal_first = False,
        stg_no_spatial = False,
        stg_no_temporal = False,
        stg_no_mlp = False,
        stg_use_pos_patch_level = False,
        stg_use_pos_frame_level = False,
        stg_use_pos_absolute = False,
        stg_use_xavier_init=False, 
        stg_use_mlp_swiglu=False, 
        stg_use_mlp_swiglu_subln=False,
        stg_use_RMSNorm=False,
        stg_use_temporal_attn_rope=False,
        stg_use_spatial_attn_rope=False,
        stg_use_rope_qkv_bias=False,
        stg_use_rope_qkv_norm=False,
        stg_no_spatial_gating = False,
        stg_no_temporal_gating = False,
        stg_no_mlp_gating = False,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.pre_encoding = pre_encoding
        self.video_input = video_input
        self.use_mask = use_mask
        self.mvm_decode = mvm_decode
        self.qformer_text_input = qformer_text_input
        self.residual_size = residual_size
        if self.video_input == 'residual':
            self.down_proj = nn.Linear(4096, 1024)
            self.non_linear_func = nn.ReLU()
            self.up_proj = nn.Linear(1024, 4096)
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

        if self.mvm_decode:
            self.mvm_decoder = Linear_Decoder()

        self.use_mlp = use_mlp
        self.use_vivit = use_vivit
        self.use_vivit_no_pos = use_vivit_no_pos
        self.use_vivit_no_spatial = use_vivit_no_spatial
        self.use_vivit_no_temporal = use_vivit_no_temporal
        
        self.use_st_attn_pre = use_st_attn_pre
        self.use_st_attn_post = use_st_attn_post
        self.use_st_attn_pos = use_st_attn_pos
        self.use_st_attn_no_spatial = use_st_attn_no_spatial
        self.use_st_attn_pre_two_layers = use_st_attn_pre_two_layers

        self.st_attn_pre_depth = st_attn_pre_depth
        self.st_attn_pre_num_heads = st_attn_pre_num_heads
        self.st_attn_post_num_heads = st_attn_post_num_heads
        
        self.use_stg_after_qformer = use_stg_after_qformer
        self.use_stg_after_vit = use_stg_after_vit
        self.use_stg_after_vitln = use_stg_after_vitln
        self.stg_num_layers = stg_num_layers
        self.stg_is_gating = stg_is_gating
        self.stg_is_temporal_first = stg_is_temporal_first
        self.stg_no_spatial = stg_no_spatial
        self.stg_no_temporal = stg_no_temporal
        self.stg_no_mlp = stg_no_mlp
        self.stg_use_pos_patch_level = stg_use_pos_patch_level
        self.stg_use_pos_frame_level = stg_use_pos_frame_level
        self.stg_use_pos_absolute = stg_use_pos_absolute
        self.stg_use_xavier_init = stg_use_xavier_init
        self.stg_use_mlp_swiglu = stg_use_mlp_swiglu
        self.stg_use_mlp_swiglu_subln = stg_use_mlp_swiglu_subln
        self.stg_use_RMSNorm = stg_use_RMSNorm
        self.stg_use_temporal_attn_rope = stg_use_temporal_attn_rope
        self.stg_use_spatial_attn_rope = stg_use_spatial_attn_rope
        self.stg_use_rope_qkv_bias = stg_use_rope_qkv_bias
        self.stg_use_rope_qkv_norm = stg_use_rope_qkv_norm
        self.stg_no_spatial_gating = stg_no_spatial_gating
        self.stg_no_temporal_gating = stg_no_temporal_gating
        self.stg_no_mlp_gating = stg_no_mlp_gating

        print('Loading VIT')
        self.vit_model = vit_model
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                if 'BTAdapter' in name:
                    continue
                param.requires_grad = False
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            if vit_model=='eva_clip_g' and not self.use_stg_after_vit and not self.use_stg_after_vitln:
                self.ln_vision = self.ln_vision.eval()
                self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        self.has_qformer = has_qformer
        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            
            if not qformer_text_input:
                self.Qformer.bert.embeddings.word_embeddings = None
                self.Qformer.bert.embeddings.position_embeddings = None
                for layer in self.Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
                self.load_from_pretrained(url_or_filename=q_former_model)
            else:
                self.Qformer.resize_token_embeddings(len(self.tokenizer))
                self.load_from_pretrained(url_or_filename=q_former_model)

            self.Qformer.cls = None
            if freeze_qformer:
                for name, param in self.Qformer.named_parameters():
                    param.requires_grad = False
                if vit_model=='eva_clip_g' and not self.use_stg_after_vit and not self.use_stg_after_vitln:
                    self.Qformer = self.Qformer.eval()
                    self.Qformer.train = disabled_train
                self.query_tokens.requires_grad = False
                logging.info("freeze Qformer")

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('Do not use Q-Former here.')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if qformer_text_input:
            self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llama_tokenizer.add_special_tokens({'bos_token': '</s>'})
            self.llama_tokenizer.add_special_tokens({'eos_token': '</s>'})
            self.llama_tokenizer.add_special_tokens({'unk_token': '</s>'})
        else:
            self.llama_tokenizer.pad_token = "$$"
        
        if self.use_mlp: # follow LLaVA1.5
            self.llama_proj = nn.Sequential(
                nn.Linear(img_f_dim, 4096),
                nn.GELU(),
                nn.Linear(4096, 4096)
            )
        elif self.use_vivit:
            vivit_num_heads = 12
            assert img_f_dim % vivit_num_heads == 0, f"img_f_dim {img_f_dim} should be divisible by vivit_num_heads {vivit_num_heads}"
            self.llama_proj = ViViT(
                frames=16,
                frame_patch_size=1,
                dim=img_f_dim,
                heads=vivit_num_heads,
                dim_head = int(img_f_dim/12),
                dropout = 0.,
                emb_dropout = 0.,
                num_query_tokens=num_query_token,
                out_dim = 4096,
                no_pos=self.use_vivit_no_pos,
                no_spatial=self.use_vivit_no_spatial,
                no_temporal=self.use_vivit_no_temporal,
            )
        else:
            # img to text, linear projector
            # img_f_dim = qformer hidden_size = 768
            # vicuna-7b-v1.1 hidden dimension = 4096
            self.llama_proj = nn.Linear(img_f_dim, 4096)

        if self.use_st_attn_pre:
            # follow bert, dim = 768, head = 12
            assert img_f_dim % self.st_attn_pre_num_heads == 0
            if self.use_st_attn_pos:
                self.pos_embedding = nn.Parameter(torch.randn(1, 16, 32, img_f_dim)) # num_frames = 16
            if not self.use_st_attn_no_spatial:
                self.spatial_attention = Attention(dim = img_f_dim, heads = self.st_attn_pre_num_heads, dim_head = int(img_f_dim/st_attn_pre_num_heads), dropout = 0.)
            self.temporal_attention = Attention(dim = img_f_dim, heads = self.st_attn_pre_num_heads, dim_head = int(img_f_dim/st_attn_pre_num_heads), dropout = 0.)
        elif self.use_st_attn_post:
            # follow vicuna, dim = 4096, head = 32
            assert 4096 % self.st_attn_post_num_heads == 0
            if self.use_st_attn_pos:
                self.pos_embedding = nn.Parameter(torch.randn(1, 16, 32, 4096)) # num_frames = 16
            if not self.use_st_attn_no_spatial:
                self.spatial_attention = Attention(dim = 4096, heads = self.st_attn_post_num_heads, dim_head = int(4096/st_attn_post_num_heads), dropout = 0.)
            self.temporal_attention = Attention(dim = 4096, heads = st_attn_post_num_heads, dim_head = int(4096/st_attn_post_num_heads), dropout = 0.)
        elif self.use_st_attn_pre_two_layers:
            # follow bert, dim = 768, head = 12
            assert img_f_dim % self.st_attn_pre_num_heads == 0
            if self.use_st_attn_pos:
                self.pos_embedding = nn.Parameter(torch.randn(1, 16, 32, img_f_dim)) # num_frames = 16
            if not self.use_st_attn_no_spatial:
                self.spatial_attention1 = Attention(dim = img_f_dim, heads = self.st_attn_pre_num_heads, dim_head = int(img_f_dim/st_attn_pre_num_heads), dropout = 0.)
                self.spatial_attention2 = Attention(dim = img_f_dim, heads = self.st_attn_pre_num_heads, dim_head = int(img_f_dim/st_attn_pre_num_heads), dropout = 0.)
            self.temporal_attention1 = Attention(dim = img_f_dim, heads = self.st_attn_pre_num_heads, dim_head = int(img_f_dim/st_attn_pre_num_heads), dropout = 0.)
            self.temporal_attention2 = Attention(dim = img_f_dim, heads = self.st_attn_pre_num_heads, dim_head = int(img_f_dim/st_attn_pre_num_heads), dropout = 0.)
        
        if self.use_stg_after_qformer:
            print("Using STG after QFormer")
            print("stg_num_layers: ", self.stg_num_layers)
            self.stg = STG(num_layers=self.stg_num_layers, 
                            hidden_dim=img_f_dim, 
                            num_heads=12, 
                            mlp_ratio=4.0, 
                            dropout=0., 
                            is_gating=self.stg_is_gating,
                            is_temporal_first=self.stg_is_temporal_first,
                            no_spatial=self.stg_no_spatial,
                            no_temporal=self.stg_no_temporal,
                            no_mlp=self.stg_no_mlp,
                            use_pos_patch_level=self.stg_use_pos_patch_level,
                            use_pos_frame_level=self.stg_use_pos_frame_level,
                            use_pos_absolute=self.stg_use_pos_absolute,
                            use_xavier_init=self.stg_use_xavier_init,
                            use_mlp_swiglu=self.stg_use_mlp_swiglu,
                            use_mlp_swiglu_subln=self.stg_use_mlp_swiglu_subln,
                            use_RMSNorm=self.stg_use_RMSNorm,
                            use_temporal_attn_rope=self.stg_use_temporal_attn_rope,
                            use_spatial_attn_rope=stg_use_spatial_attn_rope,
                            use_rope_qkv_bias=self.stg_use_rope_qkv_bias,
                            use_rope_qkv_norm=self.stg_use_rope_qkv_norm,
                            no_spatial_gating=self.stg_no_spatial_gating,
                            no_temporal_gating=self.stg_no_temporal_gating,
                            no_mlp_gating=self.stg_no_mlp_gating,

                        )
        elif self.use_stg_after_vit or self.use_stg_after_vitln:
            print("Using STG after ViT")
            print("stg_num_layers: ", self.stg_num_layers)
            self.stg = STG(num_layers=self.stg_num_layers, 
                            hidden_dim=1408, 
                            num_heads=16, 
                            mlp_ratio=4.0, 
                            dropout=0., 
                            is_gating=self.stg_is_gating,
                            is_temporal_first=self.stg_is_temporal_first,
                            no_spatial=self.stg_no_spatial,
                            no_temporal=self.stg_no_temporal,
                            no_mlp=self.stg_no_mlp,
                            use_pos_patch_level=self.stg_use_pos_patch_level,
                            use_pos_frame_level=self.stg_use_pos_frame_level,
                            use_pos_absolute=self.stg_use_pos_absolute,
                            use_xavier_init=self.stg_use_xavier_init,
                            use_mlp_swiglu=self.stg_use_mlp_swiglu,
                            use_mlp_swiglu_subln=self.stg_use_mlp_swiglu_subln,
                            use_RMSNorm=self.stg_use_RMSNorm,
                            use_temporal_attn_rope=self.stg_use_temporal_attn_rope,
                            use_spatial_attn_rope=stg_use_spatial_attn_rope,
                            use_rope_qkv_bias=self.stg_use_rope_qkv_bias,
                            use_rope_qkv_norm=self.stg_use_rope_qkv_norm,
                            no_spatial_gating=self.stg_no_spatial_gating,
                            no_temporal_gating=self.stg_no_temporal_gating,
                            no_mlp_gating=self.stg_no_mlp_gating,
                        )
            
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    def encode_img(self, image, text=None):
        device = image.device
        with self.maybe_autocast():
            T = image.shape[1]
            infer = True if len(image.shape)==4 else False
            if infer: # 'T C H W'
                print('Infer with batch size = 1')
                T = image.shape[0]
            use_image = True if T == 1 or (len(image.shape)==4) else False
            if (not use_image or len(image.shape)==5) and (self.vit_model=='eva_clip_g'): # video input
                image = einops.rearrange(image,'B T C H W -> (B T) C H W')

            image_embeds = self.visual_encoder(image)         
            
            if self.use_stg_after_vit:
                # print(f"image_embeds.shape = {image_embeds.shape}")
                assert len(image_embeds.shape) == 3, f"len(image_embeds.shape) != 3, image_embeds.shape = {image_embeds.shape}"
                _, L, D = image_embeds.shape # (B T) L D, where T = 16, L = num_patches+cls = 224^2/14^2 + 1 = 257, D = 1408 during training
                image_embeds = einops.rearrange(image_embeds,'(B T) L D -> B T L D',T=T)
                image_embeds = self.stg(image_embeds)
                image_embeds = einops.rearrange(image_embeds,'B T L D -> (B T) L D')

            image_embeds = self.ln_vision(image_embeds)
            
            if self.use_stg_after_vitln:
                # print(f"image_embeds.shape = {image_embeds.shape}")
                assert len(image_embeds.shape) == 3, f"len(image_embeds.shape) != 3, image_embeds.shape = {image_embeds.shape}"
                _, L, D = image_embeds.shape # (B T) L D, where T = 16, L = num_patches+cls = 224^2/14^2 + 1 = 257, D = 1408 during training
                image_embeds = einops.rearrange(image_embeds,'(B T) L D -> B T L D',T=T)
                image_embeds = self.stg(image_embeds)
                image_embeds = einops.rearrange(image_embeds,'B T L D -> (B T) L D')

            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                if self.qformer_text_input:
                    assert text
                    if isinstance(text, str):
                        text = [text] * query_tokens.size(0)
                    elif len(text) != query_tokens.size(0):
                        text_ = []
                        for t in text:
                            text_ += [t] * T
                        text = text_
                    text_Qformer = self.tokenizer(
                        text,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(query_tokens.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                inputs_llama = query_output.last_hidden_state[:,:query_tokens.size(1),:]
                if self.use_st_attn_pre:
                    assert len(inputs_llama.shape) == 3, f"len(inputs_llama.shape) != 3, inputs_llama.shape = {inputs_llama.shape}"
                    _, L, D = inputs_llama.shape # (B T) L D, where T = 16, L = 32, D = 768 during training
                    
                    if self.use_st_attn_pos:
                        # pos embedding, B T L D
                        inputs_llama = einops.rearrange(inputs_llama,'(B T) L D -> B T L D',T=T)
                        inputs_llama = inputs_llama + self.pos_embedding[:, :T, :L] # to handle T < 16
                        inputs_llama = einops.rearrange(inputs_llama,'B T L D -> (B T) L D',T=T)
                    
                    if not self.use_st_attn_no_spatial:
                        # spatial attention, (B T) L D
                        inputs_llama = self.spatial_attention(inputs_llama) + inputs_llama # residual connection
                    
                    inputs_llama = einops.rearrange(inputs_llama,'(B T) L D -> (B L) T D',T=T,L=L)
                    
                    # temporal attention, (B L) T D
                    inputs_llama = self.temporal_attention(inputs_llama) + inputs_llama # residual connection
                    
                    inputs_llama = einops.rearrange(inputs_llama,'(B L) T D -> (B T) L D',T=T,L=L)
                
                elif self.use_st_attn_pre_two_layers:
                    assert len(inputs_llama.shape) == 3, f"len(inputs_llama.shape) != 3, inputs_llama.shape = {inputs_llama.shape}"
                    _, L, D = inputs_llama.shape # (B T) L D, where T = 16, L = 32, D = 768 during training
                    
                    if self.use_st_attn_pos:
                        # pos embedding, B T L D
                        inputs_llama = einops.rearrange(inputs_llama,'(B T) L D -> B T L D',T=T)
                        inputs_llama = inputs_llama + self.pos_embedding[:, :T, :L] # to handle T < 16
                        inputs_llama = einops.rearrange(inputs_llama,'B T L D -> (B T) L D',T=T)
                    
                    # first layer
                    if not self.use_st_attn_no_spatial:
                        # spatial attention, (B T) L D
                        inputs_llama = self.spatial_attention1(inputs_llama) + inputs_llama # residual connection
                    inputs_llama = einops.rearrange(inputs_llama,'(B T) L D -> (B L) T D',T=T,L=L)
                    # temporal attention, (B L) T D
                    inputs_llama = self.temporal_attention1(inputs_llama) + inputs_llama # residual connection
                    inputs_llama = einops.rearrange(inputs_llama,'(B L) T D -> (B T) L D',T=T,L=L)
                    
                    # second layer
                    if not self.use_st_attn_no_spatial:
                        # spatial attention, (B T) L D
                        inputs_llama = self.spatial_attention2(inputs_llama) + inputs_llama # residual connection
                    inputs_llama = einops.rearrange(inputs_llama,'(B T) L D -> (B L) T D',T=T,L=L)
                    # temporal attention, (B L) T D
                    inputs_llama = self.temporal_attention2(inputs_llama) + inputs_llama # residual connection
                    inputs_llama = einops.rearrange(inputs_llama,'(B L) T D -> (B T) L D',T=T,L=L)
                
                if self.use_stg_after_qformer:
                    assert len(inputs_llama.shape) == 3, f"len(inputs_llama.shape) != 3, inputs_llama.shape = {inputs_llama.shape}"
                    _, L, D = inputs_llama.shape # (B T) L D, where T = 16, L = 32, D = 768 during training
                    inputs_llama = einops.rearrange(inputs_llama,'(B T) L D -> B T L D',T=T)
                    inputs_llama = self.stg(inputs_llama)
                    inputs_llama = einops.rearrange(inputs_llama,'B T L D -> (B T) L D')

                # projector initialized by InstructBLIP
                inputs_llama = self.llama_proj(inputs_llama)
                
                if self.use_st_attn_post:
                    assert len(inputs_llama.shape) == 3, f"len(inputs_llama.shape) != 3, inputs_llama.shape = {inputs_llama.shape}"
                    _, L, D = inputs_llama.shape # (B T) L D, where T = 16, L = 32, D = 4096 during training
                    
                    if self.use_st_attn_pos:
                        inputs_llama = einops.rearrange(inputs_llama,'(B T) L D -> B T L D',T=T)
                        inputs_llama = inputs_llama + self.pos_embedding[:, :T, :L] # to handle T < 16
                        inputs_llama = einops.rearrange(inputs_llama,'B T L D -> (B T) L D',T=T)

                    if not self.use_st_attn_no_spatial:
                        # spatial attention, (B T) L D
                        inputs_llama = self.spatial_attention(inputs_llama) + inputs_llama # residual connection
                    
                    inputs_llama = einops.rearrange(inputs_llama,'(B T) L D -> (B L) T D',T=T,L=L)
                    
                    # temporal attention, (B L) T D
                    inputs_llama = self.temporal_attention(inputs_llama) + inputs_llama # residual connection
                    
                    inputs_llama = einops.rearrange(inputs_llama,'(B L) T D -> (B T) L D',T=T,L=L)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))
                inputs_llama = self.llama_proj(image_embeds)
            if not infer:
                inputs_llama = einops.rearrange(inputs_llama,'(B T) L D -> B T L D',T=T)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama, use_image
  
    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for each_img_embed, each_prompt in zip(img_embeds, prompts):
                p_before, p_after = each_prompt.split('<ImageHere>')
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=self.qformer_text_input).to(img_embeds.device)
                p_before_embed = self.embed_tokens(p_before_tokens.input_ids) if min(p_before_tokens.input_ids.shape) != 0 else None
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids)
                if len(each_img_embed.size())==2:
                    each_img_embed = each_img_embed[None]
                wrapped_emb = torch.cat([p_before_embed, each_img_embed, p_after_embed], dim=1) if p_before_embed is not None \
                    else torch.cat([each_img_embed, p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)
            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img
    
    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def get_residual_index(self, sample_segments, total_segments, devices):
        if hasattr(self,'residual_index'):
            return self.residual_index
        else:
            seg_size = float(total_segments) / sample_segments
            frame_indices = np.array([
            int((seg_size / 2) + np.round(seg_size * idx))
            for idx in range(sample_segments)
            ])
            frame_indices = torch.from_numpy(frame_indices).to(devices)
            self.register_buffer('residual_index', frame_indices)
            return frame_indices

    def forward(self, samples):
        image = samples["image"]
        instruction = samples["instruction_input"] if "instruction_input" in samples else None

        use_image = False
        if self.pre_encoding:
            image = image.type_as(self.llama_proj.weight)
            img_embeds = self.llama_proj(image)
            atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(image.device)
        else:
            if self.qformer_text_input:
                qformer_text_input = [it.split('Human: ')[1].split(' ###')[0] for it in instruction]
            else:
                qformer_text_input = None
            img_embeds, atts_img, use_image = self.encode_img(image, qformer_text_input)

        if not use_image:
            T = img_embeds.size(1)
        if not use_image and self.video_input == 'all':
            img_embeds = img_embeds.view(img_embeds.size(0),1,-1,img_embeds.size(-1)).contiguous()
        elif not use_image and self.video_input == 'mean':
            img_embeds = img_embeds.mean(dim=1, keepdim=True)
        elif not use_image and self.video_input == 'residual':
            residual_index = self.get_residual_index(self.residual_size,T,img_embeds.device)
            global_embeds = img_embeds.mean(dim=1, keepdim=True)
            
            local_embeds = img_embeds[:,residual_index]
            global_embeds = global_embeds.expand((-1,self.residual_size,-1,-1)).to(self.up_proj.weight.dtype)
            global_embeds = self.up_proj(self.non_linear_func(self.down_proj(global_embeds)))
            img_embeds = (local_embeds + global_embeds).view(img_embeds.size(0),1,-1,img_embeds.size(-1)).contiguous()
        else:
            pass

        B, _, L, D = img_embeds.size()
        unmask_img_embeds = None
        if not use_image and self.use_mask:
            self.img_len = L
            rate = np.random.normal(0.5, 0.1)
            mask_rate = float(np.clip(rate,0.1,0.7))
            mask = RandomMaskingGenerator(L, mask_rate, B, img_embeds.device).unsqueeze(1)
            self.mask = mask

            unmask_img_embeds = img_embeds
            unmask_atts_img = torch.ones(unmask_img_embeds.size()[:-1], dtype=torch.long).to(image.device)
            img_embeds = img_embeds[~mask].reshape(B, 1, -1, D)
            atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(image.device)
            self.mask_img_len = img_embeds.size(2)


        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, instruction)
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.llama_tokenizer.eos_token for t in samples["answer"]] if self.qformer_text_input \
            else [t + self.end_sym for t in samples["answer"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask)

        if unmask_img_embeds is not None:
            unmask_img_embeds, unmask_atts_img = self.prompt_wrap(unmask_img_embeds, unmask_atts_img, instruction)
            unmask_inputs_embeds, unmask_attention_mask, unmask_input_lens = \
            self.concat_emb_input_output(unmask_img_embeds, unmask_atts_img, to_regress_embeds, to_regress_tokens.attention_mask)

        if not self.qformer_text_input:
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                             dtype=to_regress_tokens.input_ids.dtype,
                             device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.embed_tokens(bos)
            atts_bos = atts_img[:, :1]
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, attention_mask], dim=1)
            if unmask_img_embeds is not None:
                unmask_inputs_embeds = torch.cat([bos_embeds, unmask_inputs_embeds], dim=1)
                unmask_attention_mask = torch.cat([atts_bos, unmask_attention_mask], dim=1)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                       dtype=torch.long).to(image.device).fill_(-100)
        )

        offset = 0 if self.qformer_text_input else 1
        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + offset:input_lens[i] + len(target) + offset] = target  # plus 1 for bos

        if unmask_img_embeds is None:
            unmask_inputs_embeds, unmask_attention_mask = None, None
        return inputs_embeds, attention_mask, unmask_inputs_embeds, unmask_attention_mask, targets

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)

        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        pre_encoding = cfg.get("pre_encoding", False)
        video_input = cfg.get("video_input", None)
        use_mask = cfg.get("use_mask", False)
        qformer_text_input = cfg.get("qformer_text_input", False)
        residual_size = cfg.get("residual_size", 4)
        mvm_decode = cfg.get("mvm_decode", False)
        
        use_mlp = cfg.get("use_mlp", False)
        use_vivit = cfg.get("use_vivit", False)
        use_vivit_no_pos = cfg.get("use_vivit_no_pos", False)
        use_vivit_no_spatial = cfg.get("use_vivit_no_spatial", False)
        use_vivit_no_temporal = cfg.get("use_vivit_no_temporal", False)
        
        use_st_attn_pre = cfg.get("use_st_attn_pre", False)
        use_st_attn_post = cfg.get("use_st_attn_post", False)
        use_st_attn_pos = cfg.get("use_st_attn_pos", False)
        use_st_attn_no_spatial = cfg.get("use_st_attn_no_spatial", False)
        use_st_attn_pre_two_layers = cfg.get("use_st_attn_pre_two_layers", False)
        
        st_attn_pre_num_heads = cfg.get("st_attn_pre_num_heads", 12)
        st_attn_post_num_heads = cfg.get("st_attn_post_num_heads", 32)
        
        use_stg_after_qformer = cfg.get("use_stg_after_qformer", False)
        use_stg_after_vit = cfg.get("use_stg_after_vit", False)
        use_stg_after_vitln = cfg.get("use_stg_after_vitln", False)
        stg_num_layers = cfg.get("stg_num_layers", -1)
        stg_is_gating = cfg.get("stg_is_gating", False)
        stg_is_temporal_first = cfg.get("stg_is_temporal_first", False)
        stg_no_spatial = cfg.get("stg_no_spatial", False)
        stg_no_temporal = cfg.get("stg_no_temporal", False)
        stg_no_mlp = cfg.get("stg_no_mlp", False)
        stg_use_pos_patch_level = cfg.get("stg_use_pos_patch_level", False)
        stg_use_pos_frame_level = cfg.get("stg_use_pos_frame_level", False)
        stg_use_pos_absolute = cfg.get("stg_use_pos_absolute", False)
        stg_use_xavier_init = cfg.get("stg_use_xavier_init", False)
        stg_use_mlp_swiglu = cfg.get("stg_use_mlp_swiglu", False)
        stg_use_mlp_swiglu_subln = cfg.get("stg_use_mlp_swiglu_subln", False)
        stg_use_RMSNorm = cfg.get("stg_use_RMSNorm", False)
        stg_use_temporal_attn_rope = cfg.get("stg_use_temporal_attn_rope", False)
        stg_use_spatial_attn_rope = cfg.get("stg_use_spatial_attn_rope", False)
        stg_use_rope_qkv_bias = cfg.get("stg_use_rope_qkv_bias", False)
        stg_use_rope_qkv_norm = cfg.get("stg_use_rope_qkv_norm", False)
        stg_no_spatial_gating = cfg.get("stg_no_spatial_gating", False)
        stg_no_temporal_gating = cfg.get("stg_no_temporal_gating", False)
        stg_no_mlp_gating = cfg.get("stg_no_mlp_gating", False)

        # print(f"config = {cfg.config}")
        
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            pre_encoding=pre_encoding,
            video_input=video_input,
            use_mask=use_mask,
            mvm_decode=mvm_decode,
            residual_size=residual_size,
            qformer_text_input=qformer_text_input,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            use_mlp=use_mlp,
            use_vivit=use_vivit,
            use_vivit_no_pos=use_vivit_no_pos,
            use_vivit_no_spatial=use_vivit_no_spatial,
            use_vivit_no_temporal=use_vivit_no_temporal,
            use_st_attn_pre = use_st_attn_pre,
            use_st_attn_post = use_st_attn_post,
            use_st_attn_pos = use_st_attn_pos,
            use_st_attn_no_spatial = use_st_attn_no_spatial,
            use_st_attn_pre_two_layers = use_st_attn_pre_two_layers,
            st_attn_pre_num_heads = st_attn_pre_num_heads,
            st_attn_post_num_heads = st_attn_post_num_heads,
            use_stg_after_qformer = use_stg_after_qformer,
            use_stg_after_vit = use_stg_after_vit,
            use_stg_after_vitln = use_stg_after_vitln,
            stg_num_layers = stg_num_layers,
            stg_is_gating = stg_is_gating,
            stg_is_temporal_first = stg_is_temporal_first,
            stg_no_spatial = stg_no_spatial,
            stg_no_temporal = stg_no_temporal,
            stg_no_mlp = stg_no_mlp,
            stg_use_pos_patch_level = stg_use_pos_patch_level,
            stg_use_pos_frame_level = stg_use_pos_frame_level,
            stg_use_pos_absolute = stg_use_pos_absolute,
            stg_use_xavier_init = stg_use_xavier_init,
            stg_use_mlp_swiglu = stg_use_mlp_swiglu,
            stg_use_mlp_swiglu_subln = stg_use_mlp_swiglu_subln,
            stg_use_RMSNorm = stg_use_RMSNorm,
            stg_use_temporal_attn_rope = stg_use_temporal_attn_rope,
            stg_use_spatial_attn_rope = stg_use_spatial_attn_rope,
            stg_use_rope_qkv_bias = stg_use_rope_qkv_bias,
            stg_use_rope_qkv_norm = stg_use_rope_qkv_norm,
            stg_no_spatial_gating = stg_no_spatial_gating,
            stg_no_temporal_gating = stg_no_temporal_gating,
            stg_no_mlp_gating = stg_no_mlp_gating,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path and not os.path.isdir(ckpt_path):
            print("STLLMModel Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if 'model' in ckpt:
                ckpt = ckpt['model']
            if 'llm_proj.weight' in ckpt:
                # print("STLLMModel 'llm_proj.weight' in ckpt")
                # print("Shape of 'llm_proj.weight':", ckpt['llm_proj.weight'].shape)
                # print("Shape of 'llm_proj.bias':", ckpt['llm_proj.bias'].shape)
                # Shape of 'llm_proj.weight': torch.Size([4096, 768])
                # Shape of 'llm_proj.bias': torch.Size([4096])
                ckpt['llama_proj.weight'] = ckpt.pop('llm_proj.weight')
                ckpt['llama_proj.bias'] = ckpt.pop('llm_proj.bias')
            msg = model.load_state_dict(ckpt, strict=False)
            for name in ckpt:
                if name in model.state_dict():
                    print(f"STLLMModel Parameter {name} initialized from ckpt.")

        return model
