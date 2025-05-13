export CUDA_VISIBLE_DEVICES="5,6"
export HF_ENDPOINT="https://hf-mirror.com"

image_dir="/home/zhanghaotian/EZ-HOI/hicodet/hico_20160224_det/images/test2015/"
hdf5_file="/home/zhanghaotian/data129/hicodet_mask/test2015_mask.h5"

# image_dir="/home/zhanghaotian/EZ-HOI/hicodet/hico_20160224_det/images/train2015/"
# hdf5_file="/home/zhanghaotian/data129/hicodet_mask/train2015_mask.h5"

python -u blip_semantic_hdf5.py --image_dir $image_dir --hdf5_file $hdf5_file
