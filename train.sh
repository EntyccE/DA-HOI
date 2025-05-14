export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ENDPOINT="https://hf-mirror.com"
export SD_Config="./StableDiffusion/v1-inference.yaml"
export SD_ckpt="./StableDiffusion/v1-5-pruned-emaonly.ckpt"

mask_embedding_type=embedding
upsample_factor=1.0
mask_locate_type=label # merged
vision_decoder_layers=4
dec_layers=$(( vision_decoder_layers ))
diff_cross_attn=caption
prefix_length=8
conjun_length=2
merge_mode=alter # add
image_sd_resolution=256 # 512
min_lr=1e-7
sched=step  # warmupcos
lr_drop=60

python -m torch.distributed.launch --nproc_per_node=2 --master_port 3994 --use_env main.py \
    --batch_size 64 \
    --output_dir output129/mergeall-sched_${sched}-lr_drop_${lr_drop}-mlr_${min_lr}-newalter-${merge_mode}-${image_sd_resolution}-${diff_cross_attn}-prompt_${prefix_length}_${conjun_length}-layer_${vision_decoder_layers}_${dec_layers}-${mask_embedding_type}_up-${upsample_factor}-${mask_locate_type} \
    --epochs 80 \
    --lr 1e-4 --min-lr $min_lr --sched $sched --lr_drop $lr_drop \
    --hoi_token_length 25 \
    --enable_dec \
    --image_sd_resolution $image_sd_resolution \
    --merge_mode $merge_mode \
    --diff_cross_attn $diff_cross_attn \
    --prefix_length $prefix_length \
    --conjun_length $conjun_length \
    --vision_decoder_layers $vision_decoder_layers \
    --dec_layers $dec_layers \
    --dataset_file hico \
    --multi_scale false  --set_cost_hoi_type 0 --use_aux_text false \
    --f_idxs 5 8 11 \
    --mask_embedding_type $mask_embedding_type \
    --mask_width 1280 \
    --upsample_factor $upsample_factor \
    --mask_locate_type $mask_locate_type \
    --enable_focal_loss --description_file_path hico_hoi_descriptions.json \
    # --resume /home/zhanghaotian/MemHOI/output129/mergeall-sched_warmupcos-lr_drop_60-mlr_1e-6-newalter-alter-256-caption-prompt_8_2-layer_4_4-embedding_up-1.0-label/checkpoint.pth
    # --resume /home/zhanghaotian/MemHOI/output129/newdiff-newalter-alter-256-caption-prompt_8_2-layer_4_4-diffusion_up-1.0-label/checkpoint.pth


# Training time 6:58:15
# 10:48:20

