from typing import Optional, Tuple, Literal, List
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from diffusers import AutoPipelineForText2Image, DDIMScheduler

def build_ldm_from_cfg(model_key: str,
                       device: int = 0):
    print('Loading SD model')
    device = torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')

    pipe = AutoPipelineForText2Image.from_pretrained(model_key, torch_dtype=torch.float16).to(device)

    pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            )
        
    print('SD model loaded')
    return pipe, device

class LdmExtractor(nn.Module):

    # LDM_CONFIGS = {
    #     "SSD-1B": ("segmind/SSD-1B", "XL"),
    #     "SSD-vega": ("segmind/Segmind-Vega", "XL"),
    #     "SD1.4": ("CompVis/stable-diffusion-v1-4", None)
    # }
    LDM_CONFIGS = {
        "SSD-1B": ("hfckpts/SSD-1B", "XL"),
        "SSD-vega": ("hfckpts/Segmind-Vega", "XL"),
        "SD1.4": ("hfckpts/stable-diffusion-v1-4", None)
    }

    def __init__(
        self,
        model_name: str = "SSD-1B",
        device: int = 0, 
    ):

        super().__init__()

        self.model_name = model_name
        model_key, sd_version = self.LDM_CONFIGS[self.model_name]

        self.text_encoders = []
        self.pipe, self.device = build_ldm_from_cfg(model_key, device)
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoders.append(self.pipe.text_encoder)

        if sd_version == "XL":
            self.text_encoders.append(self.pipe.text_encoder_2)

        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(50)
    
    def register_hooks(self):
        hook_handles = []
        if self.model_name == "SD1.4":
            attn_block = self.unet.down_blocks[-2].attentions[-1].transformer_blocks[-1].attn1
        else:
            attn_block = self.unet.down_blocks[-1].attentions[-1].transformer_blocks[-1].attn1

        # def hook_self_attn(mod, input, output):
        #     self._features = output.detach()
        # hook_handles.append(attn_block.register_forward_hook(partial(hook_self_attn)))
        last_resnet = self.unet.up_blocks[0].resnets[-1]

        def hook_fn(module, inp, out):
            # out.shape will be (batch, 1280, H, W)
            features_1280 = out
            # store it somewhere, or process directly
            print("captured feature:", features_1280.shape)

        def hook_attn(mod, inputs, output):
            # import pdb; pdb.set_trace()
            # Unpack the inputs passed to the forward function.
            hidden_states = inputs[0]
            encoder_hidden_states = inputs[1] if len(inputs) > 1 and inputs[1] is not None else hidden_states
            # Optionally, inputs[2] could be the attention_mask (if provided)
            attention_mask = inputs[2] if len(inputs) > 2 else None

            # Compute query and key using the module's linear projections:
            query = mod.to_q(hidden_states)
            key = mod.to_k(encoder_hidden_states)

            # Reshape to merge head dimensions:
            query = mod.head_to_batch_dim(query)
            key = mod.head_to_batch_dim(key)
            
            # Compute the attention map (i.e. the attention probabilities)
            attention_map = mod.get_attention_scores(query, key, attention_mask)

            # Save both the final output features and the attention map.
            self._features = output.detach()
            self._attn_map = attention_map.detach()

        hook_handles.append(attn_block.register_forward_hook(partial(hook_attn)))
        # hook_handles.append(self.unet.down_blocks[-1].attentions[-1].transformer_blocks[-1].attn2.register_forward_hook(partial(hook_attn)))
        hook = last_resnet.register_forward_hook(hook_fn)
        
        return hook_handles

    def do_classifier_free_guidance(self, guidance_scale):
        return guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @torch.no_grad()
    def get_text_embeds(self, prompt, num_images_per_prompt=1, guidance_scale=1.0, img_size=1024):
        do_classifier_free_guidance = self.do_classifier_free_guidance(guidance_scale)
        batch_size = len(prompt)

        with torch.autocast(self.device.type, dtype=torch.float32):
            prompt_embeds_tuple = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance)

        if len(prompt_embeds_tuple) == 2:
            prompt_embeds, negative_prompt_embeds = prompt_embeds_tuple
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            return prompt_embeds, None

        else:
            (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ) = prompt_embeds_tuple

            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self.pipe._get_add_time_ids(
                (img_size, img_size), (0, 0), (img_size, img_size), dtype=prompt_embeds.dtype, \
                    text_encoder_projection_dim=self.text_encoders[1].config.projection_dim)
            negative_add_time_ids = add_time_ids

            if self.do_classifier_free_guidance(guidance_scale):
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(self.device)
            add_text_embeds = add_text_embeds.to(self.device)
            add_time_ids = add_time_ids.to(self.device).repeat(batch_size * num_images_per_prompt, 1)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            return prompt_embeds, added_cond_kwargs

    @torch.no_grad()
    def encode_to_latent(self, input_image):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            input_image = 2 * input_image - 1
            posterior = self.vae.encode(input_image).latent_dist
            latent_image = posterior.mean * self.vae.config.scaling_factor
        return latent_image

    def forward(self,
                img,
                num_images_per_prompt: int = 1,
                guidance_scale: float = 1.,
                step: Tuple[int, ...] = 50,
                img_size: int = 1024,
                prompts: List[str] = None
                ):

        batch_size = img.shape[0]
        images = F.interpolate(img, size=(img_size, img_size), mode='bilinear')
        batch_size = images.shape[0]

        rng = torch.Generator(device=self.device).manual_seed(42)
        if prompts is None:
            prompts = [""] * batch_size

        prompt_embeds, added_cond_kwargs = self.get_text_embeds(prompts, num_images_per_prompt, guidance_scale, img_size)

        latent_image = self.encode_to_latent(images)

        noise = torch.randn(1, 4, img_size//8, img_size//8, generator=rng, device=self.device)
        noise = noise.expand_as(latent_image)

        hook_handles = self.register_hooks()

        t = torch.tensor([step], device=self.device).expand(batch_size)
        
        noisy_latent_image = self.pipe.scheduler.add_noise(latent_image, noise, t)

        if self.do_classifier_free_guidance(guidance_scale):
            noisy_latent_image = torch.cat([noisy_latent_image] * 2)

        t = t.repeat(noisy_latent_image.shape[0] // t.shape[0])

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                noisy_latent_image = self.pipe.scheduler.scale_model_input(noisy_latent_image, t)
                self.unet(noisy_latent_image, t, encoder_hidden_states=prompt_embeds, \
                            added_cond_kwargs=added_cond_kwargs).sample
                
        for hook_handle in hook_handles:
            hook_handle.remove()

        return self._features, self._attn_map

    def get_noun_cross_attention_features(self, image, caption, caption_nouns, device):
        """
        Given an ldm_extractor instance, an image tensor, a caption string,
        and a list of noun words from that caption, this function:
        1. Tokenizes the caption using the diffusion model’s tokenizer.
        2. Finds (approximate) token indices corresponding to each noun.
        3. Registers a hook on the cross–attention module (in the last transformer block)
            so that during the forward pass the module returns a tensor with shape 
            (B, num_heads, T, S), where S is the number of text tokens.
        4. Runs a forward pass with the full caption (so that the prompt embedding 
            and cross–attention are computed only once).
        5. For each noun, extracts a slice along the S (text token) dimension.
        
        Returns:
        noun_features: dict mapping noun -> feature tensor (shape e.g. (B, num_heads, T, 1))
        token_list: the tokenized caption (for debugging / verification)
        """
        # 1. Tokenize caption using the pipeline’s tokenizer.
        tokens = self.pipe.tokenizer(caption, return_tensors="pt")
        input_ids = tokens.input_ids[0].tolist()
        token_list = self.pipe.tokenizer.convert_ids_to_tokens(input_ids)
        
        # 2. Find token indices for each noun (a simple heuristic: look for tokens that contain the noun).
        noun_token_indices = []
        for noun in caption_nouns:
            # This simple check may return multiple indices per noun if subword tokenization is used.
            # Here we take the first occurrence.
            indices = [i for i, tok in enumerate(token_list) if noun.lower() in tok.lower()]
            if indices:
                noun_token_indices.append(indices[0])
            else:
                noun_token_indices.append(None)
                print(f"Warning: noun '{noun}' not found in tokenized caption.")
        
        # Filter out any nouns that weren’t found.
        valid_nouns = []
        valid_indices = []
        for noun, idx in zip(caption_nouns, noun_token_indices):
            if idx is not None:
                valid_nouns.append(noun)
                valid_indices.append(idx)
        
        # 3. Register a hook on the cross-attention module.
        # (We choose the cross-attn of the last transformer block in the last down-block.
        # For SD1.4 models, the structure might be slightly different.)
        hook_handles = []
        if self.model_name == "SD1.4":
            cross_attn_module = self.unet.down_blocks[-2].attentions[-1].transformer_blocks[-1].attn2
        else:
            cross_attn_module = self.unet.down_blocks[-1].attentions[-1].transformer_blocks[-1].attn2

        cross_attention_data = {}  # will hold the hooked output

        def hook_cross_attn(module, input, output):
            # Here we assume that the module’s forward returns a tensor of shape (B, num_heads, T, S)
            # where the last dimension indexes the text tokens.
            cross_attention_data["output"] = output.detach()

        handle = cross_attn_module.register_forward_hook(hook_cross_attn)
        hook_handles.append(handle)
        
        # 4. Forward pass using the full caption.
        # Prepare prompt embeddings using the caption.
        prompt_embeds, added_cond_kwargs = self.get_text_embeds(
            [caption], num_images_per_prompt=1, guidance_scale=1.0, img_size=image.shape[-1]
        ) # 1, 77, 2048
        latent_image = self.encode_to_latent(image) # 1, 4, 57, 80
        noise = torch.randn_like(latent_image)
        t = torch.tensor([50], device=device).expand(latent_image.shape[0])
        noisy_latent_image = self.pipe.scheduler.add_noise(latent_image, noise, t)
        if self.do_classifier_free_guidance(1.0):
            noisy_latent_image = torch.cat([noisy_latent_image, noisy_latent_image])
            t = t.repeat(2)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _ = self.unet(noisy_latent_image, t,
                                    encoder_hidden_states=prompt_embeds,
                                    added_cond_kwargs=added_cond_kwargs).sample # 1, 4, 57, 80

        # Remove hook(s)
        for h in hook_handles:
            h.remove()
        
        cross_features = cross_attention_data.get("output", None)
        if cross_features is None:
            print("No cross-attention features captured.")
            return None, token_list
        
        # 5. Extract per-noun features.
        # Here cross_features has shape (B, token_seq_len, feature_dim), so we slice along dim 1.
        noun_features = {}
        for noun, token_idx in zip(valid_nouns, valid_indices):
            # Extract the feature at the token index. The result has shape (B, feature_dim).
            noun_feat = cross_features[:, token_idx, :]
            # Optionally, unsqueeze to add a token dimension: shape becomes (B, 1, feature_dim).
            noun_feat = noun_feat.unsqueeze(1)
            noun_features[noun] = noun_feat
        
        return noun_features, token_list