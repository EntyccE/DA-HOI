"""
SWiG-HOI dataset utils.
"""
import os
import json
import torch
import torch.utils.data
from torchvision.datasets import CocoDetection
import datasets.transforms as T
from PIL import Image
from .swig_v1_categories import SWIG_INTERACTIONS, SWIG_ACTIONS, SWIG_CATEGORIES
from utils.sampler import repeat_factors_from_category_frequency, get_dataset_indices

# NOTE: Replace the path to your file
SWIG_ROOT = "./data/swig_hoi/images_512"
SWIG_TRAIN_ANNO = "./data/swig_hoi/annotations/swig_trainval_1000.json"
SWIG_VAL_ANNO = "./data/swig_hoi/annotations/swig_test_1000.json"
SWIG_TEST_ANNO = "./data/swig_hoi/annotations/swig_test_1000.json"


class SWiGHOIDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, image_set, repeat_factor_sampling, eval_subset):
        self.root = img_folder
        self.transforms = transforms
        # Text description of human-object interactions
        dataset_texts, text_mapper = prepare_dataset_text(image_set, eval_subset=eval_subset)
        self.dataset_texts = dataset_texts
        self.text_mapper = text_mapper
        # Load dataset
        repeat_factor_sampling = repeat_factor_sampling and image_set == "train"
        reverse_text_mapper = {v: k for k, v in text_mapper.items()}
        self.dataset_dicts = load_swig_json(ann_file, img_folder, reverse_text_mapper, repeat_factor_sampling)

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
        aux_classes = torch.tensor(annos["aux_classes"], dtype=torch.int64)

        target = {
            "image_id": torch.tensor(image_id),
            "orig_size": torch.tensor([h, w]),
            "boxes": boxes,
            "classes": classes,
            "aux_classes": aux_classes,
            "hois": annos["hois"],
            "filename": filename,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.dataset_dicts)

key_idxs = [x for x in range(20)] 
key_idxs = list(set(key_idxs))
print(key_idxs)

def load_swig_json(json_file, image_root, text_mapper, repeat_factor_sampling=False):
    """
    Load a json file with HOI's instances annotation.

    Args:
        json_file (str): full path to the json file in HOI instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        text_mapper (dict): a dictionary to map text descriptions of HOIs to contiguous ids.
        repeat_factor_sampling (bool): resampling training data to increase the rate of tail
            categories to be observed by oversampling the images that contain them.
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
                'aux_classes': list[list], # n x 3, a list of auxiliary object annotations
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
    HOI_MAPPER = {(x["action_id"], x["object_id"]): x["id"] for x in SWIG_INTERACTIONS}

    imgs_anns = json.load(open(json_file, "r"))

    dataset_dicts = []
    images_without_valid_annotations = []
    for anno_dict in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, anno_dict["file_name"])
        record["height"] = anno_dict["height"]
        record["width"] = anno_dict["width"]
        record["image_id"] = anno_dict["img_id"]

        if len(anno_dict["box_annotations"]) == 0 or len(anno_dict["hoi_annotations"]) == 0:
            images_without_valid_annotations.append(anno_dict)
            continue

        boxes = [obj["bbox"] for obj in anno_dict["box_annotations"]]
        classes = [obj["category_id"] for obj in anno_dict["box_annotations"]]
        aux_classes = []
        for obj in anno_dict["box_annotations"]:
            aux_categories = obj["aux_category_id"]
            while len(aux_categories) < 3:
                aux_categories.append(-1)
            aux_classes.append(aux_categories)

        for hoi in anno_dict["hoi_annotations"]:
            target_id = hoi["object_id"]
            object_id = classes[target_id]
            action_id = hoi["action_id"]
            hoi["text"] = generate_text(action_id, object_id)
            continguous_id = HOI_MAPPER[(action_id, object_id)]
            if continguous_id not in text_mapper.keys():
                continue
            hoi["hoi_id"] = text_mapper[continguous_id]
        ## for subset evaluation, remove irrelevant images and annotations
        anno_dict["hoi_annotations"] = [hoi for hoi in anno_dict["hoi_annotations"] if "hoi_id" in hoi.keys()]
        if len(anno_dict["hoi_annotations"]) == 0:
            continue
        targets = {
            "boxes": boxes,
            "classes": classes,
            "aux_classes": aux_classes,
            "hois": anno_dict["hoi_annotations"],
        }

        record["annotations"] = targets
        dataset_dicts.append(record)
    
    if repeat_factor_sampling:
        repeat_factors = repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh=0.0001)
        dataset_indices = get_dataset_indices(repeat_factors)
        dataset_dicts = [dataset_dicts[i] for i in dataset_indices]

    return dataset_dicts


def generate_text(action_id, object_id):
    act = SWIG_ACTIONS[action_id]["name"]
    obj = SWIG_CATEGORIES[object_id]["name"]
    act_def = SWIG_ACTIONS[action_id]["def"]
    obj_def = SWIG_CATEGORIES[object_id]["def"]
    obj_gloss = SWIG_CATEGORIES[object_id]["gloss"]
    obj_gloss = [obj] + [x for x in obj_gloss if x != obj]
    if len(obj_gloss) > 1:
        obj_gloss = " or ".join(obj_gloss)
    else:
        obj_gloss = obj_gloss[0]

    # s = [act, obj_gloss]
    s = [act, obj]
    return s


''' deprecated, text
# def generate_text(action_id, object_id):
#     act = SWIG_ACTIONS[action_id]["name"]
#     obj = SWIG_CATEGORIES[object_id]["name"]
#     act_def = SWIG_ACTIONS[action_id]["def"]
#     obj_def = SWIG_CATEGORIES[object_id]["def"]
#     obj_gloss = SWIG_CATEGORIES[object_id]["gloss"]
#     obj_gloss = [obj] + [x for x in obj_gloss if x != obj]
#     if len(obj_gloss) > 1:
#         obj_gloss = " or ".join(obj_gloss)
#     else:
#         obj_gloss = obj_gloss[0]
#     # s = f"A photo of a person {act} with object {obj}. The object {obj} means {obj_def}."
#     # s = f"a photo of a person {act} with object {obj}"
#     # s = f"A photo of a person {act} with {obj}. The {act} means to {act_def}."
#     s = f"A photo of a person {act} with {obj_gloss}. The {act} means to {act_def}."
#     return s
'''


def prepare_dataset_text(image_set, eval_subset=False):
    texts = []
    text_mapper = {}
    for i, hoi in enumerate(SWIG_INTERACTIONS):
        if image_set != "train" and hoi["evaluation"] == 0: continue
        if image_set != "train" and i not in key_idxs and eval_subset: continue
        action_id = hoi["action_id"]
        object_id = hoi["object_id"]
        s = generate_text(action_id, object_id)
        text_mapper[len(texts)] = i
        texts.append(s)
    return texts, text_mapper


def make_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])

    # scales = [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
    scales = [224, 256, 288, 320]

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
            # T.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4]),
            T.RandomSelect(
                T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                T.Compose([
                    T.RandomCrop_InteractionConstraint((0.7, 0.7), 0.9),
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


''' deprecated (Fixed image resolution + random cropping + centering)
def make_transforms(image_set):

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
                    T.RandomCrop_InteractionConstraint((0.8, 0.8), 0.9),
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
'''


def build(image_set, args):
    # NOTE: Replace the path to your file
    PATHS = {
        "train": (SWIG_ROOT, SWIG_TRAIN_ANNO),
        "val": (SWIG_ROOT, SWIG_VAL_ANNO),
        "dev": (SWIG_ROOT, SWIG_TEST_ANNO),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = SWiGHOIDetection(
        img_folder,
        ann_file,
        transforms=make_transforms(image_set, args),
        image_set=image_set,
        repeat_factor_sampling=args.repeat_factor_sampling,
        eval_subset=args.eval_subset,
    )

    return dataset