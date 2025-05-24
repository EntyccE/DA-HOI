import os, pickle
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import json, pdb, math
from typing import Tuple, Union
from collections import OrderedDict
from omegaconf import OmegaConf
import gc

import utils.box_ops as box_ops
import clip
from clip.model import Transformer, LayerNorm, MLP, QuickGELU
from clip.clip import _download
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from .position_encoding import PositionEmbeddingSine
from .matcher import build_matcher
from .criterion import SetCriterion
from torchvision.ops import batched_nms
from .transformer import TransformerDecoderLayer, TransformerDecoder
from .origin_clip import VisionTransformer
from ldm.util import instantiate_from_config
from .sd_helper import UNetWrapper

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


class HOIResidualAttentionBlock(nn.Module):
    '''
    [CLS + PATCH], [HOI] attention block in HOI Vision Encoder:
        - [CLS + PATCH] x [CLS + PATCH]: original attention uses CLIP's pretrained weights.
        - [HOI] x [PATCH]: cross-attention between [HOI] tokens and image patches.
        - [HOI] x [CLS + HOI]: HOI sequential parsing.
    '''
    def __init__(self, d_model: int, n_head: int, parse_attn_mask: torch.Tensor = None, two_feat: bool = True):
        super().__init__()

        # self.hoi_parse_attn = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.hoi_self_attn = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.hoi_cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.two_feat = two_feat
        if self.two_feat:
            self.hoi_cross_attn_d = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        # self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.parse_attn_mask = parse_attn_mask

        self.hoi_ln1 = LayerNorm(d_model)
        self.hoi_ln2 = LayerNorm(d_model)
        self.hoi_ln3 = LayerNorm(d_model)
        if self.two_feat:
            self.hoi_ln_d = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        if self.two_feat:
            self.dropout_d = nn.Dropout(0.1)

    def forward(self, image: torch.Tensor, hoi: torch.Tensor, feature: torch.Tensor = None, mask: torch.Tensor = None, prompt_hint: torch.Tensor = torch.zeros(0,768)):
        # Self-attention block
        self_attn_output, _ = self.hoi_self_attn(hoi, hoi, hoi, attn_mask=self.parse_attn_mask[1:, 1:].to(hoi.device))
        hoi = hoi + self.dropout1(self_attn_output)
        hoi = self.hoi_ln1(hoi)

        # Cross-attention block
        cross_attn_output, attn_map = self.hoi_cross_attn(hoi, image, image, key_padding_mask=mask)
        hoi = hoi + self.dropout2(cross_attn_output)
        hoi = self.hoi_ln2(hoi)

        if self.two_feat and feature is None:
            raise ValueError("If two feature to decode, the second feature should not be None")

        if self.two_feat and feature is not None:
            cross_attn_output_d, attn_map_d = self.hoi_cross_attn_d(hoi, feature, feature, key_padding_mask=mask)
            hoi = hoi + self.dropout_d(cross_attn_output_d)
            hoi = self.hoi_ln_d(hoi)
        else:
            attn_map_d = None
        
        # Feed-forward block
        ff_output = self.mlp(hoi)
        hoi = hoi + self.dropout3(ff_output)
        hoi = self.hoi_ln3(hoi)

        return image, feature, hoi, attn_map, attn_map_d
    

class HOITransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None, use_mask=True, use_map=True):
        super().__init__()
        self.width = width
        self.layers = layers
        self.use_mask = use_mask
        self.use_map = use_map
        self.resblocks = nn.Sequential(*[HOIResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        
        if self.use_mask:
            self.mask_mlp = nn.Sequential(OrderedDict([
                ("mask_fc1", nn.Linear(width, width)),
                ("mask_gelu", QuickGELU()),
                ("mask_fc2", nn.Linear(width, width))
            ]))
            self.mask_ln = LayerNorm(width)
        if self.use_map:
            self.attn_mlp = nn.Sequential(OrderedDict([
                ("attn_fc1", nn.Linear(width, width)),
                ("attn_gelu", QuickGELU()),
                ("attn_fc2", nn.Linear(width, width))
            ]))
            self.attn_ln = LayerNorm(width)
            self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.mask_mlp + self.attn_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, image: torch.Tensor, df: torch.Tensor, mf: torch.Tensor, attn_image: torch.Tensor, hoi: torch.Tensor, mask: torch.Tensor = None, prompt_hint: torch.Tensor = torch.zeros(0,768)):
        for layer_i, resblock in enumerate(self.resblocks[:2]):
            image, df, hoi, attn_map, attn_map2 = resblock(image, hoi, df, mask, prompt_hint)

        image, df, hoi_t, attn_map, attn_map2 = self.resblocks[2](image, hoi, df, mask, prompt_hint)
        if self.use_mask:
            _, mf, hoi2, attn_map, attn_map2 = self.resblocks[-2](image, hoi, mf, mask, prompt_hint)
            hoi = hoi_t + self.mask_mlp(hoi2)
        else:
            hoi = hoi_t

        image, df, hoi_t, attn_map, attn_map2 = self.resblocks[3](image, hoi, df, mask, prompt_hint)
        if self.use_map:
            _, attn_image, hoi1, attn_map, attn_map2 = self.resblocks[-1](image, hoi, attn_image, mask, prompt_hint)
            hoi = hoi_t + self.attn_mlp(hoi1)
        else:
            hoi = hoi_t

        return image, hoi, attn_map


class HOIVisionTransformer(nn.Module):
    """ This module encodes RGB images and outputs HOI bounding box predictions and projected
        feature vectors in joint vision-and-text feature space.
    """
    def __init__(
        self,
        # vision backbone
        image_resolution: int,
        patch_size: int,
        width: int, # 768 # feature dim
        layers: int, # 4
        heads: int,
        output_dim: int,
        hoi_token_length: int = 5, # 25
        hoi_parser_attn_mask: torch.Tensor = None,
        region_aware_encoder_mask: torch.Tensor = None,
        # bounding box head
        enable_dec: bool = False,
        dec_heads: int = 8,
        dec_layers: int = 6,
        merge_mode: str = "add",
        use_mask: bool = True,
        use_map: bool = True,
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.hoi_token_length = hoi_token_length
        self.output_dim = output_dim
        self.patch_size = patch_size
        # Weights in original CLIP model.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = LayerNorm(width)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # Modified Transformer blocks
        self.transformer = HOITransformer(width, layers, heads, hoi_parser_attn_mask, use_mask, use_map)

        # Additional parameters for HOI detection
        self.hoi_token_embed = nn.Parameter(scale * torch.randn(hoi_token_length, width)) # 25, 768
        self.hoi_pos_embed = nn.Parameter(scale * torch.randn(hoi_token_length, width))

        # Additional parameters for detection head
        self.enable_dec = enable_dec
        if enable_dec: # True
            self.image_patch_pos = nn.Parameter(scale * torch.randn((self.image_resolution // self.patch_size) ** 2, width))
            self.hoi_parser_attn_mask = hoi_parser_attn_mask
            decoder_layer = TransformerDecoderLayer(width, dec_heads, normalize_before=True)
            decoder_norm = LayerNorm(width)
            self.bbox_head = TransformerDecoder(decoder_layer, dec_layers, decoder_norm, True)

        self.bbox_score = nn.Linear(width, 1)
        self.bbox_embed = MLP(width, width, 8, 3)
 
        self.hoi_mlp = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(width, width*2)),
            ("gelu", QuickGELU()),
            ("fc2", nn.Linear(width*2, width))
        ]))
        self.df_mlp = nn.Sequential(OrderedDict([
            ("df_fc1", nn.Linear(width, width*2)),
            ("df_gelu", QuickGELU()),
            ("df_fc2", nn.Linear(width*2, width))
        ]))

        self.hoi_ln = LayerNorm(width)
        self.df_ln = LayerNorm(width)

        self.merge_mode = merge_mode
        self.use_mask = use_mask
        self.use_map = use_map
        if merge_mode == "add":
            # used for gated fusion
            self.mf_mlp2 = nn.Sequential(OrderedDict([
                ("mf2_fc1", nn.Linear(width, width))
            ]))
            self.mf_ln2 = LayerNorm(width)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.bbox_score.weight, gain=1)
        nn.init.constant_(self.bbox_score.bias, 0)

        for layer in self.bbox_embed.layers:
            nn.init.xavier_uniform_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

    def interpolate_pos_embedding(self, x, mask):
        """ Using fixed positional embedding to handle the changing image resolution.
        Refer to https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L174
        """
        ori_h = (~mask).cumsum(1, dtype=torch.float32)[:, -1, 0]
        ori_w = (~mask).cumsum(2, dtype=torch.float32)[:, 0, -1]
        ori_shapes = [(int(h), int(w)) for h, w in zip(ori_h, ori_w)]
        bs, h, w = mask.shape

        npatch = x.shape[0] - 1
        dim = x.shape[1]

        class_pos_embed = x[0, :]
        patch_pos_embed = x[1:, :]

        w0, h0 = w // self.patch_size, h // self.patch_size
        interploated_pos_embed = torch.zeros(bs, h0, w0, dim).type_as(x)
        # Add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        for i, (hi, wi) in enumerate(ori_shapes):
            w0, h0 = wi // self.patch_size, hi // self.patch_size
            w0, h0 = w0 + 0.1, h0 + 0.1
            interploated = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(npatch)), int(math.sqrt(npatch)), dim).permute(0, 3, 1, 2),
                scale_factor=(h0 / math.sqrt(npatch), w0 / math.sqrt(npatch)),
                mode='bicubic',
            )
            assert int(h0) == interploated.shape[-2] and int(w0) == interploated.shape[-1]
            interploated = interploated.permute(0, 2, 3, 1)
            interploated_pos_embed[i, :int(h0), :int(w0), :] = interploated

        interploated_pos_embed = interploated_pos_embed.view(bs, -1, dim)
        interploated_pos_embed = torch.cat([class_pos_embed + torch.zeros(bs, 1, dim).type_as(x), interploated_pos_embed], dim=1)
        return interploated_pos_embed

    def forward(self, image: torch.Tensor, df: torch.Tensor, mf: torch.Tensor, mattn_m: torch.Tensor, mask: torch.Tensor = None, prompt_hint: torch.Tensor = torch.zeros(0,768)):
        """
        image: feature from CLIP vision
        # mf: stands for mask feature, feature extracted from mask. 
        """
        bs, num_of_grids, c = image.shape # bs, 196, 768
        hoi = self.hoi_token_embed + torch.zeros(bs, self.hoi_token_length, c).type_as(image) # TODO error multi scale
        # if not self.semantic_query: ## if use semantic query, add position embedding later
        hoi = hoi + self.hoi_pos_embed

        hoi = self.ln_pre(hoi)
        hoi = hoi.permute(1, 0, 2)  # NLD -> LND (25, bs, 768)
        image = image.permute(1, 0, 2)  # [grid ** 2, bs, width]
        df = df.permute(1, 0, 2)

        image = image + self.hoi_mlp(self.hoi_ln(image))
        df = df + self.df_mlp(self.df_ln(df))

        if self.use_mask:
            mf = mf.permute(1, 0, 2)
        if self.use_map:
            mattn_m = mattn_m.permute(1, 0, 2)

        if self.merge_mode == "alter":
            image, hoi, attn_map = self.transformer(image, df, mf, mattn_m, hoi, mask=None, prompt_hint=prompt_hint)
        elif self.merge_mode == "add":
            image = image + self.mf_ln2(self.mf_mlp2(mf))
            image, hoi, attn_map = self.transformer(image, image, mattn_m, hoi, mask=None, prompt_hint=prompt_hint)

        image = image.permute(1, 0, 2)  # LND -> NLD
        hoi = hoi.permute(1, 0, 2)  # LND -> NLD

        hoi_features = self.ln_post(hoi)
        hoi_features = hoi_features @ self.proj # [bs, 25, out_dim=512]
        # Bounding box head
        if self.enable_dec: # True
            patch_pos = self.image_patch_pos.unsqueeze(0) + torch.zeros(bs, num_of_grids, c).type_as(image) # Add zeros: stack (196, 768) to (bs, 196, 768)
            patch_pos = patch_pos.permute(1, 0, 2).type_as(image) # # 196, 64, 768
            
            hoi = hoi.permute(1, 0, 2) # NLD -> LND
            image = image.permute(1, 0, 2) # NLD -> LND

            hidden = self.bbox_head(
                tgt=hoi, # 25, bs, 768
                tgt_mask=self.hoi_parser_attn_mask[1:, 1:].to(hoi.device), # exclude [CLS] (25, 25) all zero
                query_pos=self.hoi_pos_embed[:, None, :], # 25, 1, 768
                memory=image, # 196, bs, 768
                pos=patch_pos) # hidden: [4, 25, bs, 768], 4: layers

            box_scores = self.bbox_score(hidden) # [layers, L, N, 1], 768 -> 1
            pred_boxes = self.bbox_embed(hidden).sigmoid() # [layers, L, N, 8]
            box_scores = box_scores.permute(0, 2, 1, 3) # [layers, N, L, 1]
            pred_boxes = pred_boxes.permute(0, 2, 1, 3) # [layers, N, L, 8]

            return_dict = {#"image_features": image_features,
                           "hoi_features": hoi_features,
                           "pred_boxes": pred_boxes[-1],
                           "box_scores": box_scores[-1],
                           "attn_maps": attn_map,
                        #    "aux_outputs": aux_outputs
                           }
        else:
            box_scores = self.bbox_score(hoi)
            pred_boxes = self.bbox_embed(hoi).sigmoid()
            return_dict = {#"image_features": image_features,
                           "hoi_features": hoi_features,
                           "pred_boxes": pred_boxes,
                           "box_scores": box_scores,
                           "attn_maps": attn_map}
        return return_dict


class MyStableDiffusion(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        config_path = os.environ.get("SD_Config")
        ckpt_path = os.environ.get("SD_ckpt")
        config = OmegaConf.load(config_path)
        config.model.params.ckpt_path = ckpt_path
        # config.model.params.cond_stage_config.target = 'ldm.modules.encoders.modules.AbstractEncoder'
        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model
        self.cond_stage_model = sd_model.cond_stage_model
        unet_config = dict()
        self.unet = UNetWrapper(sd_model.model, **unet_config)
        sd_model.model = None
        sd_model.first_stage_model = None
        sd_model.cond_stage_model = None
        del self.encoder_vq.decoder

        self.encoder_vq.eval()
        self.cond_stage_model.eval()
        self.unet.eval()
        self.device = device

        self.set_device()

    def set_device(self):
        for name, module in self.named_children():
            module.to(self.device)
            for param in module.parameters():
                param.requires_grad = False

    def get_learned_conditioning(self, c):
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            xc = self.cond_stage_model.encode(c)
            if isinstance(xc, DiagonalGaussianDistribution):
                xc = xc.mode()
        else:
            xc = self.cond_stage_model(c)
        return xc

    def extract_sd_feature(self, resized_img, condition, ori_device, batch_size=8):
        resized_img = resized_img.to(self.device)
        condition = condition.to(self.device)
        
        device = resized_img.device
        total = resized_img.shape[0]
        feats = []

        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = start + batch_size
                imgs_batch = resized_img[start:end]
                t = torch.zeros((imgs_batch.shape[0],), device=device, dtype=torch.long)
                latents = self.encoder_vq.encode(imgs_batch).mode().detach() # torch.Size([2, 4, 28, 28])
                cond = [condition[start:end]]
                outs = self.unet(latents, t, c_crossattn=cond)
                # collect the second‐to‐last output (cross‐attention on CLIP features)
                feats.append(outs[-2])

        res = torch.cat(feats, dim=0)
        return res.to(ori_device)
    
    def forward(self, resized_img_sd, caption_list, diff_cross_attn):
        bs = resized_img_sd.shape[0]
        ori_device = resized_img_sd.device
        if diff_cross_attn == "caption":
            diff_cond = self.get_learned_conditioning(caption_list)
        else:
            diff_cond = self.get_learned_conditioning([""] * bs)

        diffuse_feat = self.extract_sd_feature(resized_img_sd, diff_cond, ori_device) # (bs, 1280, 16, 16)

        gc.collect()
        torch.cuda.empty_cache()
        return diffuse_feat



class HOIDetector(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        image_sd_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        hoi_token_length: int,
        clip_preprocess: bool,
        vision_decoder_layers: int,
        vision_decoder_heads: int,
        # mask
        mask_width: int,
        mask_embedding_type: str,
        mask_locate_type: str,
        upsample_factor: float,
        upsample_method: str,
        downsample_method: str,
        ## multi-level
        multi_scale: bool,
        f_idxs : list,
        reverse_level_id: bool,
        # detection head
        enable_dec: bool,
        dec_heads: int,
        dec_layers: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        prefix_length: int = 8,
        conjun_length: int = 4,
        ## use_aux_text
        use_aux_text: bool = False,
        auxiliary_prefix_length: int = 4,
        use_prompt_hint: bool = False,
        merge_mode: str = "add",
        use_mask: bool = True,
        use_map: bool = True,
        # hyper params
        dataset_file: str = "",
        eval : bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.context_length = context_length
        self.hoi_token_length = hoi_token_length
        self.prompt_hint_length = 0

        # Vision
        vision_heads = vision_width // 64
        self.clip_preprocess= clip_preprocess
        self.embed_dim = embed_dim
        self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        self.vision_proj = nn.Sequential(OrderedDict([
            ("vision_proj_fc1", nn.Linear(vision_width, vision_width)),
            ("vision_proj_gelu1", QuickGELU()),
            ("vision_proj_dropout1", nn.Dropout(0.2)),
            ("vision_proj_fc2", nn.Linear(vision_width, vision_width)),
            ("vision_proj_dropout2", nn.Dropout(0.2)),
        ]))
        self.diff_proj = nn.Sequential(OrderedDict([
            ("diff_proj_fc1", nn.Linear(1280, max(vision_width, 1280))),
            ("diff_proj_gelu1", QuickGELU()),
            ("diff_proj_dropout1", nn.Dropout(0.2)),
            ("diff_proj_fc2", nn.Linear(max(vision_width, 1280), vision_width)),
            ("diff_proj_dropout2", nn.Dropout(0.2)),
        ]))

        if use_mask:
            self.mask_proj = nn.Sequential(OrderedDict([
                ("mask_proj_fc1", nn.Linear(512, max(vision_width, 512))),
                ("mask_proj_gelu1", QuickGELU()),
                ("mask_proj_dropout1", nn.Dropout(0.2)),
                ("mask_proj_fc2", nn.Linear(max(vision_width, 512), vision_width)),
                ("mask_proj_dropout2", nn.Dropout(0.2)),
            ]))
        if use_map:
            self.attn_proj = nn.Sequential(OrderedDict([
                ("attn_proj_fc1", nn.Linear(vision_width, vision_width)),
                ("attn_proj_gelu1", QuickGELU()),
                ("attn_proj_dropout1", nn.Dropout(0.2)),
                ("attn_proj_fc2", nn.Linear(vision_width, vision_width)),
                ("attn_proj_dropout2", nn.Dropout(0.2)),
            ]))

        self.gate_weight = torch.nn.Parameter(torch.as_tensor(0.0))
        # self.vision_mlp = nn.Parameter((vision_width ** -0.5) * torch.randn(vision_width, vision_width))
        self.multi_scale = multi_scale
        self.reverse_level_id = reverse_level_id
        self.f_idxs = f_idxs
        self.input_resolution = image_resolution
        self.input_sd_resolution = image_sd_resolution
        self.vision_width = vision_width

        self.hoi_visual_decoder = HOIVisionTransformer(
            image_resolution=int(image_resolution*upsample_factor),
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_decoder_layers,
            heads=vision_decoder_heads,
            output_dim=embed_dim,
            hoi_token_length=hoi_token_length,
            hoi_parser_attn_mask=self.build_hoi_attention_mask(),
            # region_aware_encoder_mask = self.build_region_aware_encoder_mask(tgt_len=hoi_token_length, mem_len=(image_resolution//vision_patch_size)**2),
            enable_dec=enable_dec,
            dec_heads=dec_heads,
            dec_layers=dec_layers,
            merge_mode=merge_mode,
            use_mask=use_mask,
            use_map=use_map
        )

        # Text
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.auxiliary_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.prefix_length = prefix_length
        self.conjun_length = conjun_length
        self.use_aux_text = use_aux_text
        self.auxiliary_prefix_length = auxiliary_prefix_length
        self.hoi_prefix = nn.Parameter(torch.empty(prefix_length, transformer_width))
        self.hoi_conjun = nn.Parameter(torch.empty(conjun_length, transformer_width))
        if auxiliary_prefix_length > 0:
            self.auxiliary_hoi_prefix = nn.Parameter(torch.empty(auxiliary_prefix_length, transformer_width))
        self.promp_proj = nn.Sequential(OrderedDict([
            ("proj_fc1", nn.Linear(embed_dim, vision_width)),
            ("proj_gelu", QuickGELU()),
            ("proj_fc2", nn.Linear(vision_width, vision_width))
        ]))
        self.use_prompt_hint = use_prompt_hint
        self.use_mask = use_mask
        self.use_map = use_map

        self.mask_embedding_type = mask_embedding_type
        self.mask_locate_type = mask_locate_type
        self.upsample_factor = upsample_factor
        self.upsample_method = upsample_method
        self.downsample_method = downsample_method
        self.merge_mode = merge_mode

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

        nn.init.normal_(self.hoi_prefix, std=0.01)
        nn.init.normal_(self.hoi_conjun, std=0.01)
        
        nn.init.normal_(self.promp_proj.proj_fc1.weight, std=0.01)
        nn.init.normal_(self.promp_proj.proj_fc2.weight, std=0.01)
        # nn.init.xavier_normal_(self.promp_proj.proj_fc2.weight)
        nn.init.normal_(self.vision_proj.vision_proj_fc1.weight, std=0.01)
        nn.init.normal_(self.vision_proj.vision_proj_fc2.weight, std=0.01)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_hoi_attention_mask(self):
        # lazily create causal attention mask, similar to text encoder
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.hoi_token_length + 1, self.hoi_token_length + 1)
        mask.fill_(0.0)
        # mask.fill_(float("-inf"))
        # mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_region_aware_encoder_mask(self, tgt_len, mem_len=196):
        mask = torch.empty(tgt_len, mem_len)
        mask.fill_(float("-inf"))
        region_len = mem_len // tgt_len
        for k in range(tgt_len):
            mask[k, k*region_len: min((k+1)*region_len, mem_len)] = 0
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def get_embedded_mask(self, mask, all_embeddings, size, option=None, downsample_method=None, mask_info=None):
        if downsample_method is None:
            downsample_method = self.downsample_method
        if option is None:
            option = self.mask_locate_type
        
        if option not in ["label", "merged"]:
            raise ValueError(f'The option must be either "label" or "merged". now {option}')

        if option == "merged":
            bs, _, h, w = mask.shape
            C = all_embeddings.shape[1]
            merged_embedding_map = torch.zeros(bs, C, h, w, device=mask.device, dtype=all_embeddings.dtype)

            for i, info in enumerate(mask_info):
                noun_mapping = info['noun_mapping'].to(mask.device)   # dict: candidate index -> index in all_embeddings
                mask_logits = info['mask_logits']       # dict: key (str(mask_id)) -> logits tensor
                # Get the mask for this image and compute unique mask IDs and inverse indices.
                mask_i = mask[i, 0]  # shape (h, w)
                unique_ids, inverse = torch.unique(mask_i, return_inverse=True)  # unique_ids: (num_unique,)
                
                merged_emb_list = []
                # Loop over the unique mask IDs (typically few per image).
                for m_id in unique_ids:
                    m_id_int = int(m_id.item())
                    if str(m_id_int) in mask_logits:
                        logits = torch.from_numpy(mask_logits[str(m_id_int)]).to(mask.device)
                        probs = torch.softmax(logits, dim=0)
                        keep = probs > 0.1
                        if keep.sum() == 0:
                            # Fallback: if none exceed threshold, take the highest prob candidate.
                            cand_indices = torch.argmax(probs, dim=0).unsqueeze(0)
                            cand_probs = probs[cand_indices]
                        else:
                            cand_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
                            cand_probs = probs[cand_indices]
                        # Map candidate indices to indices in all_embeddings using noun_mapping.
                        mapping_indices = torch.tensor(
                            [noun_mapping[int(idx.item())] for idx in cand_indices],
                            device=all_embeddings.device
                        )
                        candidate_emb = all_embeddings[mapping_indices]  # (num_candidates, C)
                        norm_probs = cand_probs / cand_probs.sum()
                        # Compute the weighted average (merged embedding) for this mask id.
                        merged_vector = (norm_probs.unsqueeze(1) * candidate_emb).sum(dim=0)
                    else:
                        # Fallback: if no logits provided for this mask, use the direct embedding.
                        merged_vector = all_embeddings[m_id_int]
                    merged_emb_list.append(merged_vector)
                
                # Stack merged embeddings for each unique mask ID into one tensor.
                merged_emb_tensor = torch.stack(merged_emb_list, dim=0)  # shape: (num_unique, C)
                # Use the inverse indices from torch.unique to assign each pixel its merged embedding.
                # This avoids iterating over each pixel.
                merged_image = merged_emb_tensor[inverse].view(h, w, C).permute(2, 0, 1)  # shape: (C, h, w)
                merged_embedding_map[i] = merged_image

            # Downsample the merged embedding map similar to the "label" branch.
            if downsample_method not in ["bilinear", "nearest", "average_pool", "max_pool"]:
                raise ValueError("upsample_method must be 'average_pool', 'max_pool', 'bilinear' or 'nearest'")
            if downsample_method == "average_pool":
                embedded_down = F.adaptive_avg_pool2d(merged_embedding_map, output_size=size)
            elif downsample_method == "mask_pool":
                embedded_down = F.adaptive_max_pool2d(merged_embedding_map, output_size=size)
            else:
                embedded_down = F.interpolate(merged_embedding_map, size=size, mode=downsample_method, align_corners=False)
            
            # Permute and reshape to (bs, h*w, C)
            embedded_down = embedded_down.permute(0, 2, 3, 1)
            embedded_down = embedded_down.view(bs, -1, C)
            return embedded_down
        
        elif option == "label":
            # mask: (bs, 1, 224, 224) where each value is an index
            # all_embeddings: (num_embeddings, embed_dim), e.g., (1352, 512)
            mask_indices = mask.squeeze(1)
            
            # Replace each index with its corresponding embedding vector.
            # This gives a tensor of shape (bs, 224, 224, 512)
            embedded = F.embedding(mask_indices, all_embeddings)
            embedded = embedded.permute(0, 3, 1, 2) # Permute to (bs, 512, 224, 224)

            if downsample_method not in ["bilinear", "nearest", "average_pool", "max_pool"]:
                raise ValueError("upsample_method must be 'average_pool', 'max_pool', 'bilinear' or 'nearest'")
        
            # TODO: decide which method
            if downsample_method == "average_pool":
                embedded_down = F.adaptive_avg_pool2d(embedded, output_size=size)
            elif downsample_method == "mask_pool":
                embedded_down = F.adaptive_max_pool2d(embedded, output_size=size)
            else:
                embedded_down = F.interpolate(embedded, size=size, mode=downsample_method, align_corners=False)
            
            # Permute back to (bs, height, width, 512)
            embedded_down = embedded_down.permute(0, 2, 3, 1)
            
            # Reshape to (bs, height*width, 512)
            bs = embedded_down.shape[0]
            embedded_down = embedded_down.view(bs, -1, all_embeddings.shape[1])
            
            return embedded_down


    def get_upsampled_feature(self, features, size, upsample_method=None):
        """
        Args:
            features (torch.Tensor): Input tensor of shape (bs, num_patches, embed_dim).
            size (tuple): Desired spatial size after upsampling, default is (28, 28).
            upsample_method (str): Upsampling method to use. Options: "bilinear" or "nearest".
        
        Returns:
            torch.Tensor: Upsampled tensor of shape (bs, height*width, embed_dim).
        """
        if upsample_method is None:
            upsample_method = self.upsample_method

        bs, num_patches, embed_dim = features.shape
        
        # Calculate the spatial dimension from the number of patches
        spatial_size = int(math.sqrt(num_patches))
        if spatial_size * spatial_size != num_patches:
            raise ValueError("The number of patches must be a perfect square.")
        
        # Reshape to (bs, spatial_size, spatial_size, embed_dim)
        features_2d = features.view(bs, spatial_size, spatial_size, embed_dim)
        
        # Permute to (bs, embed_dim, spatial_size, spatial_size) for upsampling
        features_2d = features_2d.permute(0, 3, 1, 2)
        
        if upsample_method not in ["bilinear", "nearest"]:
            raise ValueError("upsample_method must be 'bilinear' or 'nearest'")
        
        # Upsample using interpolation
        features_up = F.interpolate(features_2d, size=size, mode=upsample_method, 
                                    align_corners=False if upsample_method=="bilinear" else None)
        
        # Permute back to (bs, height, width, embed_dim)
        features_up = features_up.permute(0, 2, 3, 1)
        
        # Reshape to (bs, height*width, embed_dim)
        features_up = features_up.view(bs, -1, embed_dim)
        
        return features_up

    def _get_noun_embedding(self, all_nouns, device, context_length=10):
        token_ids = clip.tokenize(all_nouns, context_length=context_length).to(device)
        
        with torch.no_grad():
            token_features = self.token_embedding(token_ids)  # Shape: [N, L, D]
        
        pad_token = 0
        end_token = 49407
        valid_tokens = (token_ids != pad_token) & \
                    (token_ids != token_ids[:, 0].unsqueeze(1)) & \
                    (token_ids != end_token)  # Shape: [N, L]
        
        valid_mask = valid_tokens.unsqueeze(-1).float()  # Shape: [N, L, 1]
        
        summed_features = (token_features * valid_mask).sum(dim=1)  # Shape: [N, D]
        
        counts = valid_mask.sum(dim=1)  # Shape: [N, 1]
        
        avg_embeddings = summed_features / counts
        
        return avg_embeddings

    def _get_noun_feature(self, all_nouns, device):
        token_ids = clip.tokenize(all_nouns).to(device)
        
        with torch.no_grad():
            x = self.token_embedding(token_ids).type(self.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), token_ids.argmax(dim=-1)] @ self.text_projection

        return x

    def calculate_all_embeddings(self, all_nouns, option=None, device=None):
        if option is None:
            option = self.mask_embedding_type
        # Validate option.
        if option not in ["embedding", "feature", "diffusion"]:
            raise ValueError('The option must be either "embedding", "feature" or "diffusion".')
        if option == "diffusion":
            raise NotImplementedError
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if option == "embedding":
            return self._get_noun_embedding(all_nouns, device)
        elif option == "feature":
            return self._get_noun_feature(all_nouns, device)


    def encode_image(self, image, multi_scale=False, f_idxs=[]):
        return self.visual(image.type(self.dtype), multi_scale, f_idxs)

    def encode_text(self, text, pure_words=False, is_auxiliary_text=False):
        # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if is_auxiliary_text:
            x, eot_indices = self.auxiliary_texts_to_embedding(text)
        else:
            x, eot_indices = self.text_to_embedding(text, pure_words)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection

        return x

    def auxiliary_texts_to_embedding(self, auxiliary_texts, pure_words=False):
        """ text (List[List[Tensor]]): A list of action text tokens and object text tokens.
            [
                description text 1,
                description text 2,
                ...
                description text n,
            ]
        """
        all_token_embeddings = []
        eot_indices = []

        for description_token in auxiliary_texts:
            remain_length = self.context_length - self.auxiliary_prefix_length - len(description_token)
            if remain_length < 0:
                description_token = description_token[:len(description_token)+remain_length]
                remain_length = 0
                print(f"[WARNING] Input text is too long for context length {self.context_length}")
                raise RuntimeError(f"Input text is too long for context length {self.context_length}")
            eot_indices.append(self.context_length - remain_length - 1)
            padding_zeros = torch.zeros(remain_length, dtype=torch.long).to(description_token.device)
            token = torch.cat([description_token, padding_zeros])
            token_embedding = self.token_embedding(token).type(self.dtype)
            if self.auxiliary_prefix_length > 0:
                full_token_embedding = torch.cat([
                    token_embedding[0:1, :], self.auxiliary_hoi_prefix, token_embedding[1:, :]], dim=0)
            else:
                full_token_embedding = torch.cat([token_embedding[0:1, :], token_embedding[1:, :]], dim=0)
            all_token_embeddings.append(full_token_embedding)
        
        eot_indices = torch.as_tensor(eot_indices)
        x = torch.stack(all_token_embeddings, dim=0)  # [batch_size, n_ctx, d_model]
        return x, eot_indices

    def text_to_embedding(self, text, pure_words=False):
        """ text (List[List[Tensor]]): A list of action text tokens and object text tokens.
            [
                [action text 1, object text 1],
                [action text 2, object text 2],
                ...
                [action text n, object text n],
            ]
        """
        all_token_embeddings = []
        eot_indices = []
        if pure_words:
            for action_token, object_token in text:
                remain_length = self.context_length - len(action_token) - len(object_token)
                if remain_length < 0:
                    raise RuntimeError(f"Input text is too long for context length {self.context_length}")
                eot_indices.append(self.context_length - remain_length - 1)
                padding_zeros = torch.zeros(remain_length, dtype=torch.long).to(action_token.device)
                token = torch.cat([action_token, object_token, padding_zeros])
                token_embedding = self.token_embedding(token).type(self.dtype)
                all_token_embeddings.append(token_embedding)
        else:
            for action_token, object_token in text:
                remain_length = self.context_length - self.prefix_length - self.conjun_length - len(action_token) - len(object_token)
                if remain_length < 0:
                    raise RuntimeError(f"Input text is too long for context length {self.context_length}")
                eot_indices.append(self.context_length - remain_length - 1)
                padding_zeros = torch.zeros(remain_length, dtype=torch.long).to(action_token.device)
                token = torch.cat([action_token, object_token, padding_zeros])
                token_embedding = self.token_embedding(token).type(self.dtype)
                full_token_embedding = torch.cat([
                    token_embedding[0:1, :], self.hoi_prefix, token_embedding[1:len(action_token), :],
                    self.hoi_conjun, token_embedding[len(action_token):, :]], dim=0)
                all_token_embeddings.append(full_token_embedding)
        
        eot_indices = torch.as_tensor(eot_indices)
        x = torch.stack(all_token_embeddings, dim=0)  # [batch_size, n_ctx, d_model]
        return x, eot_indices

    def forward(self, image, extractor, semantic_mask, mattn_maps, text, image_mask, img_sizes, auxiliary_texts, all_noun_embeddings, mask_info, caption_list, diff_cross_attn):
        if self.use_prompt_hint:
            prompt_hint = self.encode_text(text, pure_words=True)
            prompt_hint = self.promp_proj(prompt_hint)
        else:
            prompt_hint = torch.zeros(0, self.vision_width).to(image.device)

        bs, c, h, w = image.shape
        if self.clip_preprocess:
            resized_img = [torchvision.transforms.Resize([self.input_resolution,self.input_resolution])(image[i][:, :img_sizes[i,0], :img_sizes[i,1]]) for i in range(bs)]
            resized_img_sd = [torchvision.transforms.Resize([self.input_sd_resolution,self.input_sd_resolution])(image[i][:, :img_sizes[i,0], :img_sizes[i,1]]) for i in range(bs)]
            resized_img = torch.stack(resized_img, dim=0)
            resized_img_sd = torch.stack(resized_img_sd, dim=0)
            resized_mask = [torchvision.transforms.Resize([self.input_resolution,self.input_resolution])(semantic_mask[i][:, :img_sizes[i,0], :img_sizes[i,1]]) for i in range(bs)]
            resized_mask = torch.stack(resized_mask, dim=0)
            resized_mattn = [torchvision.transforms.Resize([self.input_resolution,self.input_resolution])(mattn_maps[i][:, :img_sizes[i,0], :img_sizes[i,1]]) for i in range(bs)]
            resized_mattn = torch.stack(resized_mattn, dim=0)
            decoder_mask = None
        else:
            resized_img = torchvision.transforms.Resize([self.input_resolution,self.input_resolution])(image)
            resized_img_sd = torchvision.transforms.Resize([self.input_sd_resolution,self.input_sd_resolution])(image)
            resized_mask = torchvision.transforms.Resize([self.input_resolution,self.input_resolution])(semantic_mask)
            resized_mattn = torchvision.transforms.Resize([self.input_resolution,self.input_resolution])(mattn_maps)
            raise NotImplementedError("undefined decoder_mask")
        # vision encoder
        feature_maps = self.encode_image(resized_img, self.multi_scale, self.f_idxs)

        model_device = resized_img.device
        resized_img = resized_img.cpu()
        # if diff_cross_attn == "caption":
        #     diffuse_feat, _ = extractor(resized_img, step=50, img_size=resized_img.shape[-1], prompts=caption_list, max_batch_size=64)
        # else: 
        #     diffuse_feat, _ = extractor(resized_img, step=50, img_size=resized_img.shape[-1], max_batch_size=64)

        diffuse_feat = extractor(resized_img_sd, caption_list, diff_cross_attn)

        bs, f_sd, h_sd, w_sd = diffuse_feat.shape
        diffuse_feat = diffuse_feat.permute(0, 2, 3, 1) # (bs, h_sd, w_sd, f_sd)
        diffuse_feat = diffuse_feat.reshape(bs, h_sd * w_sd, f_sd) # (bs, h_sd*w_sd, f_sd)

        diffuse_feat = diffuse_feat.float()

        if self.merge_mode == "add":
            feature_h = int(math.sqrt(feature_maps.shape[1]))
            diffuse_h = int(math.sqrt(diffuse_feat.shape[1]))
            diffuse_feat = diffuse_feat.reshape(bs, diffuse_h, diffuse_h, -1).permute(0,3,1,2)
            diffuse_feat = F.interpolate(diffuse_feat, size=[feature_h, feature_h]).permute(0,2,3,1)
            diffuse_feat = diffuse_feat.reshape(bs, -1, diffuse_feat.shape[-1])

        # vision decoder
        if self.multi_scale:
            vision_output_lst = []
            for idx in range(len(feature_maps)):
                cur_feature_map = feature_maps[idx]
                size = int(math.sqrt(cur_feature_map.shape[1]))
                if size * size != cur_feature_map.shape[1]:
                    raise
                size = int(self.upsample_factor * int(math.sqrt(cur_feature_map.shape[1])))
                upsampled_feature_maps = self.get_upsampled_feature(feature_maps, (size, size))
                # downsampled_mask_embeddings = self.get_embedded_mask(resized_mask, all_noun_embeddings, (size, size), mask_info=mask_info)
                downsampled_mask_embeddings = diffuse_feat
                mask_maps = self.mask_proj(downsampled_mask_embeddings)
                feature_maps = self.vision_proj(upsampled_feature_maps) # torch.Size([8, 196, 768])

                vision_output = self.hoi_visual_decoder(image=feature_maps, mf=mask_maps, mask=decoder_mask, prompt_hint=prompt_hint)
                if self.reverse_level_id:
                    vision_output["level_id"] = torch.ones_like(vision_output['box_scores']) * (len(feature_maps)-idx) / max(1, len(feature_maps)-1)
                else:
                    vision_output["level_id"] = torch.ones_like(vision_output['box_scores']) * idx / max(1, len(feature_maps)-1)
                vision_output_lst.append(vision_output)
            vision_outputs = {}
            key_lst = list(vision_output_lst[0].keys())
            for k in key_lst:
                vision_outputs[k] = torch.cat([vision_output_lst[scale_i][k] for scale_i in range(len(vision_output_lst))], dim=1)
        else:
            size = int(math.sqrt(feature_maps.shape[1]))
            if size * size != feature_maps.shape[1]:
                raise
            size = int(self.upsample_factor * int(math.sqrt(feature_maps.shape[1])))
            upsampled_feature_maps = self.get_upsampled_feature(feature_maps, (size, size))
            # downsampled_mask_embeddings = self.get_embedded_mask(resized_mask, all_noun_embeddings, (size, size), mask_info=mask_info)
            # new_feature = torch.cat([upsampled_feature_maps, downsampled_mask_embeddings], dim=-1)

            downsampled_mask_embeddings = self.get_embedded_mask(resized_mask, all_noun_embeddings, (size, size), mask_info=mask_info)
            df = self.diff_proj(diffuse_feat)
            feature_maps = self.vision_proj(upsampled_feature_maps) # torch.Size([bs, 196, 768]) # MLP, 768 -> 768
            if self.use_mask:
                mask_maps = self.mask_proj(downsampled_mask_embeddings)
            else:
                mask_maps = None
            if self.use_map:
                i_size = int(math.sqrt(feature_maps.shape[1]))
                mattn_m = F.interpolate(resized_mattn, size=(i_size, i_size), mode='bilinear', align_corners=False).view(bs, 1, -1).permute(0, 2, 1)
                nan_per_image = torch.isnan(mattn_m).any(dim=0).squeeze(-1)
                if nan_per_image.any():
                    mattn_m[:, nan_per_image, :] = 1.0

                attn_image = feature_maps * mattn_m
                mattn_m = self.attn_proj(attn_image)
            else:
                mattn_m = None
            
            vision_outputs = self.hoi_visual_decoder(image=feature_maps, df=df, mf=mask_maps, mattn_m=mattn_m, mask=decoder_mask, prompt_hint=prompt_hint) # decoder

        # text encoder
        text_features = self.encode_text(text)
        if self.use_aux_text:
            auxiliary_text_features = self.encode_text(auxiliary_texts, is_auxiliary_text=True)
            auxiliary_text_features = auxiliary_text_features / auxiliary_text_features.norm(dim=-1, keepdim=True)
            auxiliary_logit_scale = self.auxiliary_logit_scale.exp()

        hoi_features = vision_outputs["hoi_features"]
        hoi_features = hoi_features / hoi_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_hoi = logit_scale * hoi_features @ text_features.t() 
        if self.use_aux_text:
            logits_per_hoi = logits_per_hoi + auxiliary_logit_scale * hoi_features @ auxiliary_text_features.t()

        return_dict = {
            "logits_per_hoi": logits_per_hoi,
            "pred_boxes": vision_outputs["pred_boxes"],
            "box_scores": vision_outputs["box_scores"],
            "attn_maps": vision_outputs["attn_maps"],
            # "level_id": vision_outputs["level_id"],
        }
        if "level_id" in vision_outputs:
            return_dict.update({"level_id": vision_outputs["level_id"]})
        if "aux_outputs" in vision_outputs:
            return_dict.update({"aux_outputs": vision_outputs["aux_outputs"]})

        return return_dict


class PostProcess(object):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, score_threshold, bbox_lambda=1, enable_softmax=False):
        self.score_threshold = score_threshold
        self.bbox_lambda = bbox_lambda
        self.enable_softmax = enable_softmax

    def __call__(self, outputs, original_size, hoi_mapper):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            original_size: For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            hoi_mapper: map the predicted classes to the hoi id specified by the dataset.
        """
        # Recover the bounding boxes based on the original image size
        pred_boxes = outputs['pred_boxes']
        pred_person_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes[:, :4])
        pred_object_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes[:, 4:])
        pred_person_boxes = pred_person_boxes.clamp(min=0, max=1)
        pred_object_boxes = pred_object_boxes.clamp(min=0, max=1)
        ori_h, ori_w = original_size
        pred_person_boxes[:, 0::2] = pred_person_boxes[:, 0::2] * ori_w
        pred_person_boxes[:, 1::2] = pred_person_boxes[:, 1::2] * ori_h
        pred_object_boxes[:, 0::2] = pred_object_boxes[:, 0::2] * ori_w
        pred_object_boxes[:, 1::2] = pred_object_boxes[:, 1::2] * ori_h

        if self.enable_softmax:
            hoi_scores = outputs['pred_logits'].softmax(dim=-1)
        else:
            hoi_scores = outputs['pred_logits'].sigmoid()
        box_scores = outputs['box_scores'].sigmoid()
        scores = hoi_scores * (box_scores ** self.bbox_lambda)

        # Filter out low confident ones
        keep = torch.nonzero(scores > self.score_threshold, as_tuple=True)
        scores = scores[keep]
        classes = keep[1]
        pred_person_boxes = pred_person_boxes[keep[0]]
        pred_object_boxes = pred_object_boxes[keep[0]]

        person_keep = batched_nms(pred_person_boxes, scores, classes, 0.5)
        object_keep = batched_nms(pred_object_boxes, scores, classes, 0.5)

        person_filter_mask = torch.zeros_like(scores, dtype=torch.bool)
        object_filter_mask = torch.zeros_like(scores, dtype=torch.bool)
        person_filter_mask[person_keep] = True
        object_filter_mask[object_keep] = True
        filter_mask = torch.logical_or(person_filter_mask, object_filter_mask)

        scores = scores[filter_mask].detach().cpu().numpy().tolist()
        classes = classes[filter_mask].detach().cpu().numpy().tolist()
        pred_boxes = torch.cat([pred_person_boxes, pred_object_boxes], dim=-1)
        pred_boxes = pred_boxes[filter_mask].detach().cpu().numpy().tolist()

        results = []
        for score, hoi_id, boxes in zip(scores, classes, pred_boxes):
            results.append([hoi_mapper[int(hoi_id)], score] + boxes)

        return results


def _get_clones(module, N):
    """ Clone a moudle N times """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        nnParams_modules = [
            "text_projection", "proj", "hoi_prefix", "hoi_conjun", "auxiliary_hoi_prefix", "hoi_pos_embed", "hoi_pos_embed2",
            "hoi_token_embed", "class_embedding", "positional_embedding", "vision_mlp", "semantic_units"]
        for name in nnParams_modules:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(args):
    ''' Build HOI detector and load pretrained CLIP weights '''
    # Build HOI detector
    model = HOIDetector(
        embed_dim=args.embed_dim,
        # vision encoder
        image_resolution=args.image_resolution, # CLIP uses fixed image resolution
        image_sd_resolution=args.image_sd_resolution, # SD uses fixed image resolution
        vision_layers=args.vision_layers,
        vision_width=args.vision_width,
        vision_patch_size=args.vision_patch_size,
        hoi_token_length=args.hoi_token_length,
        clip_preprocess=args.clip_preprocess,
        vision_decoder_layers=args.vision_decoder_layers,
        vision_decoder_heads=args.vision_decoder_heads,
        # mask
        mask_width=args.mask_width,
        mask_embedding_type=args.mask_embedding_type,
        mask_locate_type=args.mask_locate_type,
        upsample_factor=args.upsample_factor,
        upsample_method=args.upsample_method,
        downsample_method=args.downsample_method,
        # multi-level
        multi_scale=args.multi_scale,
        f_idxs = args.f_idxs,
        reverse_level_id = args.reverse_level_id,
        # bounding box head
        enable_dec=args.enable_dec,
        dec_heads=args.dec_heads,
        dec_layers=args.dec_layers,
        # text encoder
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        transformer_width=args.transformer_width,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        prefix_length=args.prefix_length,
        conjun_length=args.conjun_length,
        ## aux_text
        use_aux_text=args.use_aux_text,
        auxiliary_prefix_length=args.auxiliary_prefix_length,
        use_prompt_hint=args.use_prompt_hint,
        merge_mode=args.merge_mode,
        use_mask=args.use_mask,
        use_map=args.use_map,
        # hyper params
        dataset_file=args.dataset_file,
        eval=args.eval,
        device=torch.device(args.device),
    )

    # Load pretrained CLIP weights
    if args.clip_model in _MODELS:
        model_path = _download(_MODELS[args.clip_model], os.path.expanduser("~/.cache/clip"))
        clip_model = torch.jit.load(model_path).eval()
        # Copy the pretrained CLIP parameters as the initilized weights for our newly added modules. 
        state_dict = clip_model.state_dict()
        model.load_state_dict(state_dict, strict=False)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint["model"], strict=False)

    # Build matcher and criterion
    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.class_loss_coef, # previously, = 1
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_conf': args.conf_loss_coef,
    }
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers):
            aux_weight_dict.update({k + f'_{i}': weight_dict[k] for k in ['loss_bbox', 'loss_giou', 'loss_conf']})
            weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', "confidences"]
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        enable_focal_loss=args.enable_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        consider_all=args.consider_all,
    )
    device = torch.device(args.device)
    criterion.to(device)

    # Postprocessor for inference
    postprocessors = PostProcess(args.test_score_thresh, args.bbox_lambda, args.enable_softmax)

    return model, criterion, postprocessors