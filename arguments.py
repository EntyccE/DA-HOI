# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Additionally modified by Suchen for HOI detector
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set Human-Object Interaction Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=120, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model Setting
    parser.add_argument('--clip_model', default="ViT-B/16", type=str,
                        help="Name of pretrained CLIP model")
    ## description_file_path
    parser.add_argument('--description_file_path', default="", type=str,
                        help="Path to the hoi description file")
    # parser.add_argument('--frozen_weights', type=str, default=None,)
    # * Vision
    parser.add_argument('--embed_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)") # 768
    parser.add_argument('--image_resolution', default=224, type=int,
                        help="input image resolution to the vision transformer")
    parser.add_argument('--image_sd_resolution', default=512, type=int,
                        help="input image resolution to the diffusion model")
    parser.add_argument('--vision_layers', default=12, type=int,
                        help="number of layers in vision transformer")
    parser.add_argument('--vision_width', default=768, type=int,
                        help="feature channels in vision transformer") # 1024
    parser.add_argument('--vision_patch_size', default=16, type=int,
                        help="patch size: the input image is divided into multiple patches") # 14
    parser.add_argument('--hoi_token_length', default=5, type=int,
                        help="number of [HOI] tokens added to transformer's input")
    parser.add_argument('--clip_preprocess', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to preprocess images as CLIP')
    parser.add_argument('--vision_decoder_layers', default=4, type=int,
                        help="number of layers in vision transformer")
    parser.add_argument('--vision_decoder_heads', default=8, type=int,
                        help="number of layers in vision transformer")
    # mask
    parser.add_argument('--mask_width', default=512, type=int,
                    help="number of mask feature channels")
    parser.add_argument('--mask_embedding_type', default='embedding', type=str, choices=["embedding", "feature", "diffusion"],
                    help="method to get semantic embedding for masks")
    parser.add_argument('--mask_locate_type', default='label', type=str, choices=["label", "merged"],
                    help="method to replace mask index with semantic embeddings")
    parser.add_argument('--upsample_factor', default=1.0, type=float,
                    help="upsample factor of clip feature when concated with mask features")
    parser.add_argument('--upsample_method', default='bilinear', type=str, choices=["bilinear", "nearest"],
                        help='upsample method used for clip feature')
    parser.add_argument('--downsample_method', default='average_pool', type=str, choices=["bilinear", "nearest", "average_pool", "max_pool"],
                    help='downsample method used for mask feature')
    parser.add_argument('--diff_cross_attn', default='none', type=str, choices=["none", "caption"],
                help='cross attn used in diffusion')
    parser.add_argument('--use_mask', default=False, type=lambda x: (str(x).lower() == 'true'), help="use mask in input or not")
    parser.add_argument('--use_map', default=False, type=lambda x: (str(x).lower() == 'true'), help="use map in input or not")
    parser.add_argument('--merge_mode', default='none', type=str, choices=["alter", "add"],
                help='cross attn used in diffusion')
    
    ## multi level
    parser.add_argument('--multi_scale', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use prompt hint in the text encoder')
    parser.add_argument('--f_idxs', nargs='+', type=int)
    parser.add_argument('--reverse_level_id', default=False, type=lambda x: (str(x).lower() == 'true'))
    ## semantic query
    parser.add_argument('--semantic_query', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use prompt hint in the text encoder')
    parser.add_argument('--semantic_units_file', default="", type=str,
                        help='whether to use prompt hint in the text encoder')
    # * Text
    parser.add_argument('--context_length', default=77, type=int,
                        help="Maximum length of the text description")
    parser.add_argument('--vocab_size', default=49408, type=int,
                        help="Vocabulary size pre-trained with text encoder")
    parser.add_argument('--transformer_width', default=512, type=int,
                        help="feature channels in text tranformer")  # 768
    parser.add_argument('--transformer_heads', default=8, type=int,
                        help="number of multi-attention heads in text transformer")
    parser.add_argument('--transformer_layers', default=12, type=int,
                        help="number of layers in text transformer")
    parser.add_argument('--prefix_length', default=8, type=int,
                        help="number of [PREFIX] tokens at the beginning of sentences")
    parser.add_argument('--conjun_length', default=2, type=int,
                        help="number of [CONJUN] tokens between actions and objects")
    parser.add_argument('--use_aux_text', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--auxiliary_prefix_length', default=0, type=int)
    parser.add_argument('--use_prompt_hint', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use prompt hint in the text encoder')
    # hyper params
    parser.add_argument('--hoi_dropout_weight', default=0.1, type=float)
    parser.add_argument('--feature_map_dropout_weight', default=0.1, type=float)
    # * Bounding box head
    parser.add_argument('--enable_dec', action='store_true', help='enable decoders')
    parser.add_argument('--dec_heads', default=8, type=int,
                        help="Number of multi-head attention")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of layers in the bounding box head")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--enable_focal_loss', action='store_true', help='enable decoders')
    parser.add_argument('--focal_alpha', default=0.3, type=float)
    parser.add_argument('--focal_gamma', default=1.0, type=float)
    # * Matcher
    parser.add_argument('--set_cost_class', default=5, type=float,
                        help="class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_conf', default=10, type=float,
                        help="box confidence score coefficient in the matching cost")
    parser.add_argument('--hoi_type', default="center-dis", type=str, choices=["min-size", "max-size", "center-dis", "rel-center-dis"],
                        help="hoi_type in the matching cost")
    parser.add_argument('--set_cost_hoi_type', default=0, type=float,
                        help="hoi_type coefficient in the matching cost")
    parser.add_argument('--consider_all', default=False, type=lambda x: (str(x).lower() == 'true'))
    # * Loss coefficients
    parser.add_argument('--class_loss_coef', default=5, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--conf_loss_coef', default=10, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="relative classification weight of the no-object class")
    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                         help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                         help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                         help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # Dataset parameters
    parser.add_argument('--dataset_file', default='swig', choices=['hico', 'swig'])
    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='apply repeat factor sampling to increase the rate at which tail categories are observed')
    parser.add_argument('--zero_shot_exp', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='[specific for hico], treat 120 rare interactions as zero shot')
    parser.add_argument('--ignore_non_interaction', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='[specific for hico], ignore <non_interaction> category')
    parser.add_argument('--zero_shot_type', default="rare_first", type=str, choices=["default", "uc0", "uc1", "uc2", "uc3", "uc4",
                                            "rare_first", "non_rare_first", "unseen_object", "unseen_verb"],)
    parser.add_argument('--enable_softmax', default=True, type=lambda x: (str(x).lower() == 'true'))
    # Inference
    parser.add_argument('--test_score_thresh', default=0.0001, type=float,
                        help="threshold to filter out HOI predictions")
    parser.add_argument('--eval_size', default=448, type=int, help="image resolution for evaluation")
    parser.add_argument('--vis_outputs', action='store_true', help='visualize the model outputs')
    parser.add_argument('--vis_dir', default='', help='path where to save visualization results')
    parser.add_argument('--bbox_lambda', default=2.0, type=float)
    parser.add_argument('--aux_text_weight', default=1.0, type=float)
    parser.add_argument('--best_beta', default=1.0, type=float)
    parser.add_argument('--eval_subset', default=False, type=lambda x: (str(x).lower() == 'true'))
    # Training setup
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrained', default='', help='path to checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    # * Log and Device
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # * Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', help='url used to set up distributed training')
    parser.add_argument('--num_workers', default=2, type=int)
    return parser