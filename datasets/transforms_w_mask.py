# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random
import numpy as np
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from utils.box_ops import box_xyxy_to_cxcywh
from utils.misc import interpolate


def crop(image, mask, attn, target, region):
    ori_w, ori_h = image.size
    cropped_image = F.crop(image, *region)
    cropped_mask = F.crop(mask, *region) if mask is not None else None
    cropped_attn = F.crop(attn, *region) if attn is not None else None

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    """ deprecated, this part is mainly for ResizeAndCenterCrop (deprecated)
    # Image is padded with 0 if the crop region is out of boundary.
    # We use `image_mask` to indicate the padding regions.
    image_mask = torch.zeros((h, w)).bool()
    image_mask[:abs(i), :] = True
    image_mask[:, :abs(j)] = True
    image_mask[abs(i) + ori_h :, :]  = True
    image_mask[:, abs(j) + ori_w :]  = True
    target["image_mask"] = image_mask
    """

    fields = ["classes"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

        id_mapper = {}
        cnt = 0
        for i, is_kept in enumerate(keep):
            if is_kept:
                id_mapper[i] = cnt
                cnt += 1

        if "hois" in target:
            kept_hois = []
            for hoi in target["hois"]:
                if keep[hoi["subject_id"]] and keep[hoi["object_id"]]:
                    hoi["subject_id"] = id_mapper[hoi["subject_id"]]
                    hoi["object_id"] = id_mapper[hoi["object_id"]]
                    kept_hois.append(hoi)
            target["hois"] = kept_hois

    return cropped_image, cropped_mask, cropped_attn, target


def hflip(image, mask, attn, target):
    flipped_image = F.hflip(image)
    flipped_mask = F.hflip(mask)
    flipped_attn = F.hflip(attn)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, flipped_mask, flipped_attn, target


def resize(image, mask, attn, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    # import pdb;pdb.set_trace()
    maxs = size
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            w_mod = np.mod(w, 16)
            h_mod = np.mod(h, 16)
            h = h - h_mod
            w = w - w_mod
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
            ow_mod = np.mod(ow, 16)
            oh_mod = np.mod(oh, 16)
            ow = ow - ow_mod
            oh = oh - oh_mod
        else:
            oh = size
            ow = int(size * w / h)
            ow_mod = np.mod(ow, 16)
            oh_mod = np.mod(oh, 16)
            ow = ow - ow_mod
            oh = oh - oh_mod

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    rescaled_mask = F.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)
    rescaled_attn = F.resize(attn, size)

    if target is None:
        return rescaled_image, rescaled_mask, rescaled_attn, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, rescaled_mask, rescaled_attn, target


def pad(image, mask, attn, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    padded_mask = F.pad(mask, (0, 0, padding[0], padding[1]))
    padded_attn = F.pad(attn, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, padded_mask, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, padded_mask, padded_attn, target


def resize_long_edge(image, mask, attn, target, size, max_size=None):
    """Resize the image based on the long edge."""
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size

        if (w >= h and w == size) or (h >= w and h == size):
            return (h, w)

        if w > h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    rescaled_mask = F.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)
    rescaled_attn = F.resize(attn, size)

    if target is None:
        return rescaled_image, rescaled_mask, rescaled_attn, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, rescaled_mask, rescaled_attn, target


class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation)

    def __call__(self, img, mask, attn, target):
        return self.color_jitter(img), mask, attn, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, attn, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, mask, attn, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, mask, attn, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, mask, attn, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, attn, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, mask, attn, target, (crop_top, crop_left, crop_height, crop_width))


class ResizeAndCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, attn, target):
        img, mask, attn, target = resize_long_edge(img, mask, attn, target, self.size)
        image_width, image_height = img.size
        crop_height, crop_width = self.size, self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, mask, attn, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, attn, target):
        if random.random() < self.p:
            return hflip(img, mask, attn, target)
        return img, mask, attn, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, mask, attn, target=None):
        size = random.choice(self.sizes)
        return resize(img, mask, attn, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, mask, attn, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, mask, attn, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, mask, attn, target):
        if random.random() < self.p:
            return self.transforms1(img, mask, attn, target)
        return self.transforms2(img, mask, attn, target)


class ToTensor(object):
    def __call__(self, img, mask, attn, target):
        img = F.to_tensor(img)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if isinstance(attn, np.ndarray):
            attn = torch.from_numpy(attn)
        return img, mask, attn, target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, mask, attn, target):
        return self.eraser(img), mask, attn, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, attn, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, mask, attn, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, mask, attn, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, attn, target):
        for t in self.transforms:
            image, mask, attn, target = t(image, mask, attn, target)
        return image, mask, attn, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomCrop_InteractionConstraint(object):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that at most interactions
    in the image can be kept.
    """
    def __init__(self, crop_ratio, p: float):
        """
        Args:
            crop_type, crop_size: same as in :class:`RandomCrop`
        """
        self.crop_ratio = crop_ratio
        self.p = p

    def __call__(self, image, mask, attn, target):
        boxes = target["boxes"]
        w, h = image.size[:2]
        croph, cropw = int(h * self.crop_ratio[0]), int(w * self.crop_ratio[1])
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0_choice = np.arange(0, h - croph + 1)
        w0_choice = np.arange(0, w - cropw + 1)
        h_prob, w_prob = np.ones(len(h0_choice)), np.ones(len(w0_choice))
        for box in boxes:
            h_min = min(int(box[1] - croph) + 1, len(h_prob))
            h_max = int(box[3])
            w_min = min(int(box[0] - cropw) + 1, len(w_prob))
            w_max = int(box[2])
            if h_min > 0:
                h_prob[:h_min] = h_prob[:h_min] * self.p
            if h_max < h0_choice[-1]:
                h_prob[h_max:] = h_prob[h_max:] * self.p
            if w_min > 0:
                w_prob[:w_min] = w_prob[:w_min] * self.p
            if w_max < w0_choice[-1]:
                w_prob[w_max:] = w_prob[w_max:] * self.p
        h_prob, w_prob = h_prob / h_prob.sum(), w_prob / w_prob.sum() 
        h0 = int(np.random.choice(h0_choice, 1, p=h_prob)[0])
        w0 = int(np.random.choice(w0_choice, 1, p=w_prob)[0])
        return crop(image, mask, attn, target, (h0, w0, croph, cropw))