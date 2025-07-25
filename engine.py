# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Suchen for HOI detection
"""
Train and eval functions used in main.py
"""
import math, random
import sys, gc
from typing import Iterable
import torch, torchvision
import utils.misc as utils
from models.model import convert_weights
from datasets import build_evaluator
from utils.visualizer import Visualizer
from fvcore.nn import FlopCountAnalysis, flop_count_table
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models import HOIDetector
import torch.nn.functional as F
_tokenizer = _Tokenizer() # tokenizer here


def train_one_epoch(model: torch.nn.Module, extractor: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, noun_list: list[str],
                    device: torch.device, epoch: int, args):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    all_noun_embeddings = model.module.calculate_all_embeddings(noun_list, device=device)
    # all_noun_embeddings = None
    hoi_descriptions = get_hoi_descriptions(dataset_name=args.dataset_file, description_file_path=args.description_file_path)
    
    for images, semantic_masks, mattn_maps, targets_ in metric_logger.log_every(data_loader, print_freq, header):
        images, semantic_masks, mattn_maps, targets, mask_info, caption_list, texts, auxiliary_texts = prepare_inputs(images, semantic_masks, mattn_maps, targets_, data_loader, device, hoi_descriptions)
        if args.consider_all:
            texts, auxiliary_texts = prepare_text_inputs(model, data_loader.dataset.dataset_texts, device, hoi_descriptions)
        # images.tensors:torch.Size([8, 3, 320, 480]); images.mask: torch.Size([8, 320, 480])
        img_sizes = torch.stack([targets[z]['size'] for z in range(len(targets))], dim=0)

        outputs = model(images.tensors, extractor, semantic_masks.tensors, mattn_maps.tensors, texts, images.mask, img_sizes, auxiliary_texts, all_noun_embeddings, mask_info, caption_list, args.diff_cross_attn) # dict_keys(['logits_per_hoi', 'pred_boxes', 'box_scores', 'attn_maps', 'level_id'])
        loss_dict, indices = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    gc.collect()
    torch.cuda.empty_cache()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: HOIDetector, extractor, postprocessors, criterion, data_loader, noun_list, device, args):
    model.eval()
    criterion.eval()

    all_noun_embeddings = model.calculate_all_embeddings(noun_list, option=args.mask_embedding_type, device=device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # Convert applicable model parameters to fp16
    # convert_weights(model)

    # Build evaluator
    evaluator = build_evaluator(args)
    hoi_descriptions = get_hoi_descriptions(dataset_name=args.dataset_file, description_file_path=args.description_file_path)
    # hoi_descriptions: dict, e.g. 'feed zebra': ['The mouth is open wide.', 'The hand is reaching out.']

    # Convert all interaction categories into embeddings, only forward pass once!!
    text_tokens, auxiliary_texts = prepare_text_inputs(model, data_loader.dataset.dataset_texts, device, hoi_descriptions)
    # text_tokens: token of key in hoi_descriptions, list of list
    # auxiliary_texts: token of value in hoi_descriptions, list 
    # Above is guessing

    text_features = model.encode_text(text_tokens, pure_words=False) # (600, 512)
    text_features /= text_features.norm(dim=-1, keepdim=True) 
    if args.use_aux_text:
        auxiliary_text_features = model.encode_text(auxiliary_texts, is_auxiliary_text=True)
        auxiliary_text_features /= auxiliary_text_features.norm(dim=-1, keepdim=True)
    if args.use_prompt_hint:
        prompt_hint = model.encode_text(text_tokens, pure_words=True)
        prompt_hint = model.promp_proj(prompt_hint)
    else:
        prompt_hint = torch.zeros(0, args.vision_width).half().to(device)
    
    # Inference
    for images, semantic_masks, mattn_maps, targets_ in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device)
        semantic_masks = semantic_masks.to(device)
        mattn_maps = mattn_maps.to(device)
        targets = [{k: v.to(device) if (k != "hois" and k != "filename" and k != "mask_logits" and k != "noun_mapping" and k != "caption") else v for k, v in t.items()} for t in targets_]
        mask_info = [{k: v for k, v in t.items() if k == "noun_mapping" or k == "mask_logits"} for t in targets_]
        caption_list = [t["caption"] for t in targets_]
        # image_id, orig_size, bbox, hois ['subject_id', object_id', 'hoi_id', 'action_id', 'text'], 'size'
        bs, c, h, w = images.tensors.shape # bs, 3, 688, 736
        img_sizes = torch.stack([targets[z]['size'] for z in range(len(targets))], dim=0)
        s1 = args.image_resolution
        s2 = args.image_sd_resolution
        if args.clip_preprocess: # True
            resized_img = [torchvision.transforms.Resize([224,224])(images.tensors[i][:, :img_sizes[i,0], :img_sizes[i,1]]) for i in range(bs)]
            resized_img = torch.stack(resized_img, dim=0)
            resized_mask = [torchvision.transforms.Resize([224,224])(semantic_masks.tensors[i][:, :img_sizes[i,0], :img_sizes[i,1]]) for i in range(bs)]
            resized_mask = torch.stack(resized_mask, dim=0)
            resized_mattn = [torchvision.transforms.Resize([224,224])(mattn_maps.tensors[i][:, :img_sizes[i,0], :img_sizes[i,1]]) for i in range(bs)]
            resized_mattn = torch.stack(resized_mattn, dim=0)
            resized_img_sd = [torchvision.transforms.Resize([s2,s2])(images.tensors[i][:, :img_sizes[i,0], :img_sizes[i,1]]) for i in range(bs)]
            resized_img_sd = torch.stack(resized_img_sd, dim=0)
            # add same operation to mask here. 
            decoder_mask = None
        else:
            resized_img = torchvision.transforms.Resize([224,224])(images.tensors)
            resized_mask = torchvision.transforms.Resize([224,224])(semantic_masks.tensors)
            resized_mattn = torchvision.transforms.Resize([224,224])(resized_mattn.tensors)
            resized_img_sd = torchvision.transforms.Resize([s2,s2])(images.tensors)
            raise NotImplementedError("undefined decoder_mask")
        
        # vision encoder
        feature_maps = model.encode_image(resized_img, model.multi_scale, model.f_idxs) # (bs=64, 196, 768)
        # if args.diff_cross_attn == "caption":
        #     diff_cond = model.get_learned_conditioning(caption_list)
        # else:
        #     diff_cond = model.get_learned_conditioning([""] * bs)

        # diffuse_feat = model.extract_sd_feature(resized_img_sd, diff_cond) # (bs, 1280, 16, 16)

        diffuse_feat = extractor(resized_img_sd, caption_list, args.diff_cross_attn)

        bs, f_sd, h_sd, w_sd = diffuse_feat.shape
        diffuse_feat = diffuse_feat.permute(0, 2, 3, 1) # (bs, h_sd, w_sd, f_sd)
        diffuse_feat = diffuse_feat.reshape(bs, h_sd * w_sd, f_sd) # (bs, h_sd*w_sd, f_sd)
        diffuse_feat = diffuse_feat.float()

        if model.multi_scale: # False
            vision_output_lst = []
            for idx in range(len(feature_maps)):
                cur_feature_map = feature_maps[idx]
                size = int(math.sqrt(cur_feature_map.shape[1]))
                if size * size != cur_feature_map.shape[1]:
                    raise
                size = int(args.upsample_factor * int(math.sqrt(cur_feature_map.shape[1])))
                upsampled_feature_maps = model.get_upsampled_feature(cur_feature_map, (size, size), upsample_method=args.upsample_method)
                # downsampled_mask_embeddings = model.get_embedded_mask(resized_mask, all_noun_embeddings, (size, size), mask_info=mask_info)
                downsampled_mask_embeddings=None
                new_feature = torch.cat([upsampled_feature_maps, downsampled_mask_embeddings], dim=-1)
                vision_output = model.hoi_visual_decoder(image=new_feature, mask=decoder_mask, prompt_hint=prompt_hint)
                vision_output["level_id"] = torch.ones_like(vision_output['box_scores']) * idx / (len(feature_maps)-1)
                vision_output_lst.append(vision_output)
            vision_outputs = {}
            key_lst = list(vision_output_lst[0].keys())
            for k in key_lst:
                vision_outputs[k] = torch.cat([vision_output_lst[scale_i][k] for scale_i in range(len(vision_output_lst))], dim=1)

        else:
            size = int(math.sqrt(feature_maps.shape[1]))
            if size * size != feature_maps.shape[1]:
                raise
            size = int(args.upsample_factor * int(math.sqrt(feature_maps.shape[1])))
            upsampled_feature_maps = model.get_upsampled_feature(feature_maps, (size, size), upsample_method=args.upsample_method)
            downsampled_mask_embeddings = model.get_embedded_mask(resized_mask, all_noun_embeddings, (size, size), mask_info=mask_info)
            # new_feature = torch.cat([upsampled_feature_maps, downsampled_mask_embeddings], dim=-1)
            df = model.diff_proj(diffuse_feat)
            feature_maps = model.vision_proj(upsampled_feature_maps) # torch.Size([bs, 196, 768]) # MLP, 768 -> 768
            if args.use_mask:
                mask_maps = model.mask_proj(downsampled_mask_embeddings)
            else:
                mask_maps = None
            if args.use_map:
                i_size = int(math.sqrt(feature_maps.shape[1]))
                mattn_m = F.interpolate(resized_mattn, size=(i_size, i_size), mode='bilinear', align_corners=False).view(bs, 1, -1).permute(0, 2, 1)
                nan_per_image = torch.isnan(mattn_m).any(dim=0).squeeze(-1)
                if nan_per_image.any():
                    mattn_m[:, nan_per_image, :] = 1.0

                attn_image = feature_maps * mattn_m
                mattn_m = model.attn_proj(attn_image)
            else:
                mattn_m = None
            vision_outputs = model.hoi_visual_decoder(image=feature_maps, df=df, mf=mask_maps, mattn_m=mattn_m, mask=decoder_mask, prompt_hint=prompt_hint) # decoder

        # vision_outputs (dict): 'hoi_features' (512), 'pred_boxes' (8), 'box_scores' (1), 'attn_maps (196)'
        hoi_features = vision_outputs['hoi_features'] # (bs, 25, 512)
        hoi_features = hoi_features / hoi_features.norm(dim=-1, keepdim=True)
        
        logits_per_hoi = model.logit_scale.exp() * hoi_features @ text_features.t() # bs, 25, 600(num_class)

        if args.use_aux_text: # False
            aux_text_logits = model.auxiliary_logit_scale.exp() * hoi_features @ auxiliary_text_features.t()
            # aux_text_logits = ((-1) * (args.best_beta - args.best_beta * aux_text_logits)).exp()
            logits_per_hoi = logits_per_hoi + aux_text_logits
        
        pred_boxes = vision_outputs["pred_boxes"]
        box_scores = vision_outputs["box_scores"]

        outputs = {"logits_per_hoi": logits_per_hoi,
                   "pred_boxes": pred_boxes,
                   "box_scores": box_scores,
                #    "aux_outputs": vision_outputs["aux_outputs"],
                   "attn_maps": vision_outputs['attn_maps'],
                #    "level_id": vision_outputs["level_id"],
                   }
        if "level_id" in vision_outputs:
            outputs.update({"level_id": vision_outputs["level_id"]})
        
        loss_dict, indices = criterion(outputs, targets) # {'loss_ce', 'class_error', 'loss_bbox', 'loss_giou', 'loss_conf'}
        weight_dict = criterion.weight_dict

        if args.vis_outputs:
            visualizer = Visualizer(args)
            visualizer.visualize_preds(images, targets, outputs)
            # visualizer.visualize_attention(images, targets, outputs)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        
        results = {int(targets[i]['image_id']): postprocessors(
            {'pred_logits': logits_per_hoi[i], 'pred_boxes': pred_boxes[i], 'box_scores': box_scores[i]},
            targets[i]['orig_size'],
            data_loader.dataset.text_mapper
        ) for i in range(len(images.tensors))}

        evaluator.update(results)

        gc.collect()
        torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluator.save_preds()
    # accumulate predictions from all images
    evaluator.accumulate()
    evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.eval_subset:
        from datasets.swig import key_idxs
        import numpy as np
        print("all APs:", evaluator.swig_ap[np.asarray(key_idxs)])
        print("mean AP:", np.mean(evaluator.swig_ap[np.asarray(key_idxs)]))
    return stats, evaluator


def prepare_inputs(images, masks, mattn, targets_, data_loader, device, hoi_descriptions=None):
    """Prepare model inputs."""
    # image inputs
    images = images.to(device)
    masks = masks.to(device)
    mattn = mattn.to(device)
    targets = [{k: v.to(device) if (k != "hois" and k != "filename" and k != "mask_logits" and k != "noun_mapping" and k != "caption") else v for k, v in t.items()} for t in targets_]
    mask_info = [{k: v for k, v in t.items() if k == "noun_mapping" or k == "mask_logits"} for t in targets_]
    caption_list_ = [t["caption"] for t in targets_]
    # text inputs
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    texts = []
    auxiliary_texts = []
    text_inputs = []
    unique_hois = set()

    for t in targets:
        for hoi in t["hois"]:
            # Ensure all texts are unique (no duplicates).
            hoi_id = hoi["hoi_id"]
            if hoi_id in unique_hois:
                continue
            else:
                unique_hois.add(hoi_id)
            action_text, object_text = hoi["text"]
            
            hoi_name = " ".join(hoi["text"])
            # cur_hoi_description = random.sample(hoi_descriptions[hoi_name], len(hoi_descriptions[hoi_name]))
            cur_hoi_description = " ".join(hoi_descriptions[hoi_name])
            cur_hoi_description_token = _tokenizer.encode(cur_hoi_description)
            cur_hoi_description_token = torch.as_tensor([sot_token] + cur_hoi_description_token + [eot_token], dtype=torch.long).to(device)
            auxiliary_texts.append(cur_hoi_description_token)

            ## <action, object>
            action_token = _tokenizer.encode(action_text.replace("_", " "))
            object_token = _tokenizer.encode(object_text.replace("_", " "))

            action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
            object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
            texts.append([action_token, object_token])
            # text_inputs.append(action_text + " " + object_text)

    # caption_tokens_list = []
    # for caption in caption_list_:
    #     cur_caption_token = _tokenizer.encode(caption)
    #     cur_caption_token = torch.as_tensor([sot_token] + cur_caption_token + [eot_token], dtype=torch.long).to(device)
    #     caption_tokens_list.append(cur_caption_token)

    # [specific for HICO-DET], load related hois based on the targets in mini-batch
    if hasattr(data_loader.dataset, 'object_to_related_hois') and hasattr(data_loader.dataset, 'action_to_related_hois'):
        object_to_related_hois = data_loader.dataset.object_to_related_hois
        action_to_related_hois = data_loader.dataset.action_to_related_hois

        related_texts = []
        related_auxiliary_texts = []
        related_text_inputs = []
        unique_actions = set()
        unique_objects = set()
        unique_related_hois = set()
        for t in targets:
            for hoi in t["hois"]:
                hoi_id = hoi["hoi_id"]
                query_action_text, query_object_text = hoi["text"]
                if query_action_text in unique_actions or query_object_text in unique_objects:
                    continue
                else:
                    unique_actions.add(query_action_text)
                    unique_objects.add(query_object_text)

                related_hois = action_to_related_hois[query_action_text]
                for hoi in related_hois:
                    hoi_id = hoi["hoi_id"]
                    if hoi_id in unique_hois:
                        continue
                    if hoi_id in unique_related_hois:
                        continue
                    else:
                        unique_related_hois.add(hoi_id)

                    action_text, object_text = hoi["text"]
                    action_token = _tokenizer.encode(action_text.replace("_", " "))
                    object_token = _tokenizer.encode(object_text.replace("_", " "))
                    action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
                    object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
                    related_texts.append([action_token, object_token])
                    related_text_inputs.append(action_text + " " + object_text)
                    ## hoi descriptions
                    hoi_name = " ".join([action_text, object_text])
                    cur_hoi_description = " ".join(hoi_descriptions[hoi_name])
                    cur_hoi_description_token = _tokenizer.encode(cur_hoi_description)
                    cur_hoi_description_token = torch.as_tensor([sot_token] + cur_hoi_description_token + [eot_token], dtype=torch.long).to(device)
                    related_auxiliary_texts.append(cur_hoi_description_token)

                related_hois = object_to_related_hois[query_object_text]
                for hoi in related_hois:
                    hoi_id = hoi["hoi_id"]
                    if hoi_id in unique_hois:
                        continue
                    if hoi_id in unique_related_hois:
                        continue
                    else:
                        unique_related_hois.add(hoi_id)

                    action_text, object_text = hoi["text"]
                    action_token = _tokenizer.encode(action_text.replace("_", " "))
                    object_token = _tokenizer.encode(object_text.replace("_", " "))
                    action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
                    object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
                    related_texts.append([action_token, object_token])
                    related_text_inputs.append(action_text + " " + object_text)
                    ## hoi descriptions
                    hoi_name = " ".join([action_text, object_text])
                    cur_hoi_description = " ".join(hoi_descriptions[hoi_name])
                    cur_hoi_description_token = _tokenizer.encode(cur_hoi_description)
                    cur_hoi_description_token = torch.as_tensor([sot_token] + cur_hoi_description_token + [eot_token], dtype=torch.long).to(device)
                    auxiliary_texts.append(cur_hoi_description_token)
        texts.extend(related_texts)
        auxiliary_texts.extend(related_auxiliary_texts)

    return images, masks, mattn, targets, mask_info, caption_list_, texts, auxiliary_texts


@torch.no_grad()
def prepare_text_inputs(model, texts, device, hoi_descriptions):
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    text_tokens = []
    auxiliary_texts = []
    for action_text, object_text in texts:
        hoi_name = " ".join([action_text.replace(" ", "_"), object_text])
        # cur_hoi_description = random.sample(hoi_descriptions[hoi_name], len(hoi_descriptions[hoi_name]))
        cur_hoi_description = " ".join(hoi_descriptions[hoi_name])
        cur_hoi_description_token = _tokenizer.encode(cur_hoi_description)
        cur_hoi_description_token = torch.as_tensor([sot_token] + cur_hoi_description_token + [eot_token], dtype=torch.long).to(device)
        auxiliary_texts.append(cur_hoi_description_token)

        ## <action, object>
        action_token = _tokenizer.encode(action_text.replace("_", " "))
        object_token = _tokenizer.encode(object_text.replace("_", " "))
        action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
        object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
        text_tokens.append([action_token, object_token])

        # action_token = _tokenizer.encode(action_text.replace("_", " "))
        # object_token = _tokenizer.encode(object_text.replace("_", " "))

        # action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
        # object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
        # text_tokens.append([action_token, object_token])

    # text_features = model.encode_text(text_tokens, pure_words)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_tokens, auxiliary_texts


def get_flop_stats(model, data_loader):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.
    Returns:
        float: the total number of gflops of the given model.
    """
    inputs = _get_model_analysis_input(data_loader)
    flops = FlopCountAnalysis(model, inputs)
    print("Total FLOPs(G)", flops.total() / 1e9)
    print(flop_count_table(flops, max_depth=4, show_param_shapes=False))
    return flops


def _get_model_analysis_input(data_loader):
    for images, targets in data_loader:
        images, targets, texts = prepare_inputs(images, targets, "cuda")
        inputs = (images.tensors, texts, images.mask)
        return inputs


from datasets.swig_v1_categories import SWIG_ACTIONS, SWIG_CATEGORIES, SWIG_INTERACTIONS
from datasets.hico_categories import HICO_INTERACTIONS
import json

def get_hoi_descriptions(dataset_name, description_file_path):
    '''
    return: Dict {hoi_id: List[hoi-description1, ...]}
    '''
    res = {}
    assert dataset_name in description_file_path
    with open(description_file_path, "r") as f:
        hoi_descriptions = json.load(f)
    
    if "swig" in dataset_name:
        for hoi in SWIG_INTERACTIONS:
            res[hoi["name"]] = hoi_descriptions[hoi["name"]]
    else:
        for hoi in HICO_INTERACTIONS:
            hoi_name = " ".join([hoi["action"], hoi["object"]])
            res[hoi_name] = hoi_descriptions[hoi_name]
    return res
    