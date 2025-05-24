export CUDA_VISIBLE_DEVICES="0,1" # two gpu needed
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
min_lr=1e-6
sched=warmupcos
lr_drop=60

python -m torch.distributed.launch --nproc_per_node=1 --master_port 3990 --use_env main.py \
    --batch_size 64 \
    --output_dir output/checkpoint.pth \
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
    --eval \
    --pretrained ./resources/best_checkpoint.pth
