import torch, torchvision
from torchvision.transforms.functional import InterpolationMode
import os, pdb
import argparse
from PIL import Image
import json
import sys
from tqdm import tqdm

from arguments import get_args_parser
from datasets import build_dataset
from datasets.swig_v1_categories import SWIG_INTERACTIONS


training_categories = [x['name'] for x in SWIG_INTERACTIONS if not (x["evaluation"] == 1 and x["frequency"] == 0)]
print(f"Training categories: {len(training_categories)}")

parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
dataset_train = build_dataset(image_set='train', args=args)

SWIG_TRAIN_ANNO = "./data/swig_hoi/annotations/swig_trainval_1000.json"
with open(SWIG_TRAIN_ANNO, 'r') as f:
    train_data = json.load(f)

cnt = 0
multihoi_cnt = 0
for idx in tqdm(range(len(dataset_train)//20)):
    image, target = dataset_train[idx]
    assert train_data[idx]['file_name'] == target['filename'].split("/")[-1], f"{train_data[idx]['file_name']} != {target['filename']}"
    cnt += (len(train_data[idx]['hoi_annotations']) == len(target['hois']))
    # print(image.shape, target["hois"])
    multihoi_cnt += len(target['hois']) > 1

print(f"Matched {cnt} out of {len(dataset_train)} images.")
print(f"Images with multiple HOIs: {multihoi_cnt}")