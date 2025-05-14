"""
HICO-DET dataset utils
"""
import os
import json
import h5py
import collections
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import CocoDetection
# import datasets.transforms as T
import datasets.transforms_w_mask as T
from PIL import Image
from .hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS, ZERO_SHOT_INTERACTION_IDS, NON_INTERACTION_IDS, hico_unseen_index
from utils.sampler import repeat_factors_from_category_frequency, get_dataset_indices


# NOTE: Replace the path to your file
HICO_TRAIN_ROOT = "./data/hico_20160224_det/images/train2015"
HICO_TRAIN_ANNO = "./data/hico_20160224_det/annotations/trainval_hico_ann.json"
HICO_VAL_ROOT = "./data/hico_20160224_det/images/test2015"
HICO_VAL_ANNO = "./data/hico_20160224_det/annotations/test_hico_ann.json"

# place the path here
HICO_TRAIN_ATTN = "TRAIN ATTENTION PATH"
HICO_VAL_ATTN = "TEST ATTENTION PATH"

HICO_TRAIN_MASK = "TRAIN MASK PATH"
HICO_VAL_MASK = "TEST MASK PATH"

class HICO(CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        image_set,
        zero_shot_exp,
        repeat_factor_sampling,
        ignore_non_interaction,
        zero_shot_type,
        mask_locate_type,
        mask_path,
        attn_path,
    ):
        """
        Args:
            json_file (str): full path to the json file in HOI instances annotation format.
            image_root (str or path-like): the directory where the images in this json file exists.
            transforms (class): composition of image transforms.
            image_set (str): 'train', 'val', or 'test'.
            repeat_factor_sampling (bool): resampling training data to increase the rate of tail
                categories to be observed by oversampling the images that contain them.
            zero_shot_exp (bool): if true, see the last 120 rare HOI categories as zero-shot,
                excluding them from the training data. For the selected rare HOI categories, please
                refer to `<datasets/hico_categories.py>: ZERO_SHOT_INTERACTION_IDS`.
            ignore_non_interaction (bool): Ignore non-interaction categories, since they tend to
                confuse the models with the meaning of true interactions.
        """
        self.root = img_folder
        self.transforms = transforms
        # Text description of human-object interactions
        dataset_texts, text_mapper = prepare_dataset_text()
        self.dataset_texts = dataset_texts
        self.text_mapper = text_mapper # text to contiguous ids for evaluation
        object_to_related_hois, action_to_related_hois = prepare_related_hois(hico_unseen_index[zero_shot_type], ignore_non_interaction)
        self.object_to_related_hois = object_to_related_hois
        self.action_to_related_hois = action_to_related_hois
        # Load dataset
        repeat_factor_sampling = repeat_factor_sampling and image_set == "train"
        zero_shot_exp = zero_shot_exp and image_set == "train"
        self.dataset_dicts = load_hico_json(
            json_file=ann_file,
            image_root=img_folder,
            zero_shot_exp=zero_shot_exp,
            repeat_factor_sampling=repeat_factor_sampling,
            ignore_non_interaction=ignore_non_interaction,
            zero_shot_interaction_ids=hico_unseen_index[zero_shot_type])
        
        # self.dataset_dicts = self.dataset_dicts[:1]
        # keys = [os.path.basename(d["file_name"]) for d in self.dataset_dicts]

        self.dataset_dicts = self.dataset_dicts
        keys = None

        self.mask_path = mask_path 
        self.attn_path = attn_path
        self.mask_dict = load_h5_to_dict(self.mask_path, self.attn_path, keys)
        self.mask_locate_type = mask_locate_type
        
        self.all_nouns = self._unify_noun_idx()
        
    def _unify_noun_idx(self):
        # Build unified_nouns list and mapping dict for fast lookup.
        unified_nouns = []
        seen = set()
        for info in self.mask_dict.values():
            for noun in info["nouns"]:
                if noun not in seen:
                    seen.add(noun)
                    unified_nouns.append(noun)
        unified_dict = {noun: idx for idx, noun in enumerate(unified_nouns)}

        # Update each mask using vectorized indexing.
        for key, info in self.mask_dict.items():
            local_nouns = info["nouns"]
            
            # Create a mapping for local indices to global indices.
            # mapping[i] will be the global index for local noun at position i.
            mapping = [unified_dict[noun] for noun in local_nouns]
            mapping_tensor = torch.tensor(mapping, dtype=torch.int)

            if self.mask_locate_type == "merged":
                self.mask_dict[key]["mask"] = torch.from_numpy(info["original_mask"]).int()
            else:
                # Convert the old mask to a tensor and update it using vectorized indexing.
                old_mask = torch.from_numpy(info["mask"]).int()
                new_mask = mapping_tensor[old_mask]
                self.mask_dict[key]["mask"] = new_mask

            self.mask_dict[key]["mapping"] = mapping_tensor 

        return unified_nouns

    def __getitem__(self, idx: int):

        filename = self.dataset_dicts[idx]["file_name"]
        image = Image.open(filename).convert("RGB")

        w, h = image.size
        assert w == self.dataset_dicts[idx]["width"], "image shape is not consistent."
        assert h == self.dataset_dicts[idx]["height"], "image shape is not consistent."

        image_id = self.dataset_dicts[idx]["image_id"]
        annos = self.dataset_dicts[idx]["annotations"]

        boxes = torch.as_tensor(annos["boxes"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor(annos["classes"], dtype=torch.int64)

        imagename = os.path.basename(self.dataset_dicts[idx]["file_name"])
        mask = self.mask_dict[imagename]["mask"]
        attn = self.mask_dict[imagename]["attn_map"]
        
        target = {
            "image_id": torch.tensor(image_id),
            "orig_size": torch.tensor([h, w]),
            "boxes": boxes,
            "classes": classes,
            "hois": annos["hois"],
            "filename": imagename,
            "noun_mapping": self.mask_dict[imagename]["mapping"],
            "mask_logits": self.mask_dict[imagename]["mask_logits"],
            "caption": self.mask_dict[imagename]["caption"],
        }

        if self.transforms is not None:
            image, mask, attn, target = self.transforms(image, mask, attn, target)

        return image, mask, attn, target

    def __len__(self):
        return len(self.dataset_dicts)


# def load_h5_to_dict(hdf5_file):
#     data_dict = {}
#     with h5py.File(hdf5_file, 'r') as f:
#         # Iterate over each image group in the file.
#         for image_key in f.keys():
#             group = f[image_key]
#             # Create a dictionary for each image.
#             image_data = {}
#             image_data["mask"] = group["mask"][()]  # Semantic mask.
#             image_data["original_mask"] = group["original_mask"][()]  # Refined original mask.
#             # image_data["diffusion_feature"] = group["diffusion_feature"][()]  # Diffusion feature vector.
            
#             # Load mask_logits stored in the subgroup.
#             mask_logits = {}
#             for mask_id in group["mask_logits"].keys():
#                 mask_logits[mask_id] = group["mask_logits"][mask_id][()]
#             image_data["mask_logits"] = mask_logits
            
#             # Load attributes.
#             image_data["caption"] = group.attrs["caption"]
#             image_data["nouns"] = group.attrs["nouns"].split(',')

#             if any([len(n_) > 25 for n_ in image_data["nouns"]]):
#                 mapping = {}
#                 removed_count = 0
#                 new_nouns = []
#                 for i, noun in enumerate(image_data["nouns"]):
#                     if len(noun) > 25:
#                         # Record that this index should be mapped to 0 in the mask.
#                         mapping[i] = 0
#                         removed_count += 1
#                     else:
#                         mapping[i] = i - removed_count
#                         new_nouns.append(noun)
                
#                 # Adjust the mask:
#                 # For each pixel, if its value is one of the removed indices, the mapping returns 0;
#                 # Otherwise, subtract the number of removed indices before it.
#                 vec_map = np.vectorize(lambda x: mapping[x])
#                 adjusted_mask = vec_map(image_data["mask"])
#                 image_data["mask"] = adjusted_mask
#                 image_data["nouns"] = new_nouns
                
#             # Save this image's data into the main dict.
#             data_dict[image_key] = image_data
#     return data_dict


def load_h5_to_dict(hdf5_file, attn_file, keys=None):
    data_dict = {}
    with h5py.File(hdf5_file, 'r') as f:
        with h5py.File(attn_file, 'r') as f1:
        # Iterate over each image group in the file.
            for image_key in f.keys():
                if (keys is not None) and (image_key not in keys):
                    continue
                group = f[image_key]

                # Create a dictionary for each image.
                image_data = {}
                image_data["mask"] = group["mask"][()]  # Semantic mask.
                image_data["original_mask"] = group["original_mask"][()]  # Refined original mask.
                # image_data["diffusion_feature"] = group["diffusion_feature"][()]  # Diffusion feature vector.

                attn_group = f1[image_key.split('.')[0]]
                attn_map = attn_group['mean'][:] # 512, 512
                h1, w1 = image_data["mask"].shape[-2], image_data["mask"].shape[-1]
                attn_t = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()
                attn_t = F.interpolate(attn_t, size=(h1, w1), mode='bilinear', align_corners=False)
                attn_map = attn_t.squeeze(0)
                if torch.isnan(attn_map).any():
                    attn_map = torch.ones_like(attn_map)

                image_data["attn_map"] = attn_map
            
                # Load mask_logits stored in the subgroup.
                mask_logits = {}
                for key in group["mask_logits"].keys():
                    mask_logits[key] = group["mask_logits"][key][()]
                image_data["mask_logits"] = mask_logits
                
                # Load attributes.
                original_nouns = group.attrs["nouns"].split(',')
                # try:
                #     original_nouns = group.attrs["nouns"].split(',')
                # except:
                #     print(image_key)
                #     print(mask_logits)
                #     exit(0)

                # HICO_train2015_00013257.jpg

                image_data["caption"] = group.attrs["caption"]
                image_data["nouns"] = original_nouns
                
                # If any noun is too long, remove it and update the semantic mask and mask_logits.
                if any(len(noun) > 25 for noun in original_nouns):
                    # Get the indices of the valid (<=25 characters) nouns.
                    valid_indices = [i for i, noun in enumerate(original_nouns) if len(noun) <= 25]
                    
                    # Adjust the semantic mask:
                    # For each pixel, if its value corresponds to an invalid noun (i.e. not in valid_indices), set it to 0.
                    # Otherwise, find its new index among the valid ones.
                    mapping = {}
                    for new_idx, orig_idx in enumerate(valid_indices):
                        mapping[orig_idx] = new_idx
                    vec_map = np.vectorize(lambda x: mapping[x] if x in mapping else 0)
                    image_data["mask"] = vec_map(image_data["mask"])
                    
                    # Update nouns list with only valid ones.
                    new_nouns = [original_nouns[i] for i in valid_indices]
                    image_data["nouns"] = new_nouns
                    
                    # Update mask_logits:
                    # Iterate over all keys. Each tensor is 1D with a length equal to the number of original nouns.
                    # We remove entries corresponding to removed nouns.
                    new_mask_logits = {}
                    for key, logits in image_data["mask_logits"].items():
                        new_mask_logits[key] = logits[valid_indices]
                    image_data["mask_logits"] = new_mask_logits
                
                # Save this image's data into the main dict.
                data_dict[image_key] = image_data
    return data_dict



def load_hico_json(
    json_file,
    image_root,
    zero_shot_exp=True,
    repeat_factor_sampling=False,
    ignore_non_interaction=True,
    zero_shot_interaction_ids=[],
):
    """
    Load a json file with HOI's instances annotation.

    Args:
        json_file (str): full path to the json file in HOI instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        repeat_factor_sampling (bool): resampling training data to increase the rate of tail
            categories to be observed by oversampling the images that contain them.
        zero_shot_exp (bool): if true, see the last 120 rare HOI categories as zero-shot,
            excluding them from the training data. For the selected rare HOI categories, please
            refer to `<datasets/hico_categories.py>: ZERO_SHOT_INTERACTION_IDS`.
        ignore_non_interaction (bool): Ignore non-interaction categories, since they tend to
            confuse the models with the meaning of true interactions.
    Returns:
        list[dict]: a list of dicts in the following format.
        {
            'file_name': path-like str to load image,
            'height': 480,
            'width': 640,
            'image_id': 222,
            'annotations': {
                'boxes': list[list[int]], # n x 4, bounding box annotations
                'classes': list[int], # n, object category annotation of the bounding boxes
                'hois': [
                    {
                        'subject_id': 0,  # person box id (corresponding to the list of boxes above)
                        'object_id': 1,   # object box id (corresponding to the list of boxes above)
                        'action_id', 76,  # person action category
                        'hoi_id', 459,    # interaction category
                        'text': ('ride', 'skateboard') # text description of human action and object
                    }
                ]
            }
        }
    """
    imgs_anns = json.load(open(json_file, "r"))

    id_to_contiguous_id_map = {x["id"]: i for i, x in enumerate(HICO_OBJECTS)}
    action_object_to_hoi_id = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

    dataset_dicts = []
    images_without_valid_annotations = []
    for anno_dict in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, anno_dict["file_name"])
        record["height"] = anno_dict["height"]
        record["width"] = anno_dict["width"]
        record["image_id"] = anno_dict["img_id"]

        ignore_flag = False
        if len(anno_dict["annotations"]) == 0 or len(anno_dict["hoi_annotation"]) == 0:
            images_without_valid_annotations.append(anno_dict)
            continue

        boxes = [obj["bbox"] for obj in anno_dict["annotations"]]
        classes = [obj["category_id"] for obj in anno_dict["annotations"]]
        hoi_annotations = []
        for hoi in anno_dict["hoi_annotation"]:
            action_id = hoi["category_id"] - 1 # Starting from 1
            target_id = hoi["object_id"]
            object_id = id_to_contiguous_id_map[classes[target_id]]
            text = (HICO_ACTIONS[action_id]["name"], HICO_OBJECTS[object_id]["name"])
            hoi_id = action_object_to_hoi_id[text]

            # Ignore this annotation if we conduct zero-shot simulation experiments
            # if zero_shot_exp and (hoi_id in ZERO_SHOT_INTERACTION_IDS):
            if zero_shot_exp and hoi_id in zero_shot_interaction_ids:
                ignore_flag = True
                continue

            # Ignore non-interactions
            if ignore_non_interaction and action_id == 57:
                continue

            hoi_annotations.append({
                "subject_id": hoi["subject_id"],
                "object_id": hoi["object_id"],
                "action_id": action_id,
                "hoi_id": hoi_id,
                "text": text
            })

        if len(hoi_annotations) == 0 or ignore_flag:
            continue

        targets = {
            "boxes": boxes,
            "classes": classes,
            "hois": hoi_annotations,
        }

        record["annotations"] = targets
        dataset_dicts.append(record)

    if repeat_factor_sampling:
        repeat_factors = repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh=0.003)
        dataset_indices = get_dataset_indices(repeat_factors)
        dataset_dicts = [dataset_dicts[i] for i in dataset_indices]

    return dataset_dicts


def prepare_dataset_text():
    texts = []
    text_mapper = {}
    for i, hoi in enumerate(HICO_INTERACTIONS):
        action_name = " ".join(hoi["action"].split("_"))
        object_name = hoi["object"]
        s = [action_name, object_name]
        text_mapper[len(texts)] = i
        texts.append(s)
    return texts, text_mapper


def prepare_related_hois(zero_shot_interaction_ids ,ignore_non_interaction):
    ''' Gather related hois based on object names and action names
    Returns:
        object_to_related_hois (dict): {
            object_text (e.g., chair): [
                {'hoi_id': 86, 'text': ['carry', 'chair']},
                {'hoi_id': 87, 'text': ['hold', 'chair']},
                ...
            ]
        }

        action_to_relatedhois (dict): {
            action_text (e.g., carry): [
                {'hoi_id': 10, 'text': ['carry', 'bicycle']},
                {'hoi_id': 46, 'text': ['carry', 'bottle']},
                ...
            ]
        }
    '''
    object_to_related_hois = collections.defaultdict(list)
    action_to_related_hois = collections.defaultdict(list)

    for x in HICO_INTERACTIONS:
        action_text = x['action']
        object_text = x['object']
        hoi_id = x['interaction_id']
        if hoi_id in zero_shot_interaction_ids or (hoi_id in NON_INTERACTION_IDS and ignore_non_interaction):
            continue
        hoi_text = [action_text, object_text]

        object_to_related_hois[object_text].append({'hoi_id': hoi_id, 'text': hoi_text})
        action_to_related_hois[action_text].append({'hoi_id': hoi_id, 'text': hoi_text})

    return object_to_related_hois, action_to_related_hois


def make_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])

    scales = [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
            T.RandomSelect(
                T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                T.Compose([
                    T.RandomCrop_InteractionConstraint((0.75, 0.75), 0.8),
                    T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                ])
            ),
            normalize,
        ])

    if image_set == "val":
        return T.Compose([
            T.RandomResize([args.eval_size], max_size=args.eval_size * 1333 // 800),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')

    """ deprecated (Fixed image resolution + random cropping + centering)
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
            T.RandomSelect(
                T.ResizeAndCenterCrop(224),
                T.Compose([
                    T.RandomCrop_InteractionConstraint((0.7, 0.7), 0.9),
                    T.ResizeAndCenterCrop(224)
                ]),
            ),
            normalize
        ])
    if image_set == "val":
        return T.Compose([
            T.ResizeAndCenterCrop(224),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')
    """


def build(image_set, args):
    # NOTE: Replace the path to your file
    PATHS = {
        "train": (HICO_TRAIN_ROOT, HICO_TRAIN_ANNO, HICO_TRAIN_MASK, HICO_TRAIN_ATTN),
        "val": (HICO_VAL_ROOT, HICO_VAL_ANNO, HICO_VAL_MASK, HICO_VAL_ATTN),
    }

    img_folder, ann_file, mask_path, attn_path = PATHS[image_set]
    dataset = HICO(
        img_folder,
        ann_file,
        transforms=make_transforms(image_set, args),
        image_set=image_set,
        zero_shot_exp=args.zero_shot_exp,
        repeat_factor_sampling=args.repeat_factor_sampling,
        ignore_non_interaction=args.ignore_non_interaction,
        zero_shot_type=args.zero_shot_type,
        mask_locate_type=args.mask_locate_type,
        mask_path=mask_path,
        attn_path=attn_path,
    )

    return dataset