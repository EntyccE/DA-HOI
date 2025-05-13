import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import pickle  # no longer used for final saving
from PIL import Image
import matplotlib.pyplot as plt
import nltk
from matplotlib.colors import ListedColormap
import seaborn as sns
from transformers import BlipForConditionalGeneration, BlipProcessor
import h5py  # NEW: for HDF5 saving
import gc  # at the top of your file

# Download necessary NLTK data (only the first time)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from tools.ldm import LdmExtractor
from diffcut.recursive_normalized_cut import DiffCut
from tools.clip_classifier import CLIP, open_vocabulary, get_classification_logits
from tools.utils import MaskPooling
from tools.pamr import PAMR
from scipy.ndimage import median_filter

# ------------------------------------------------------------------------
# Device Setup:
# Move CLIP and BLIP to cuda:1, leaving the LDM extractor on cuda:0.
# ------------------------------------------------------------------------
device_clip = torch.device("cuda:1")  # For CLIP and related processing
device_blip = torch.device("cuda:1")  # For BLIP captioning

# --------------------------
# Initialize segmentation components on device_clip (cuda:1)
# --------------------------
pretrained = "/home/zhanghaotian/DiffCut/open_clip_pytorch_model.bin"
model_name = "convnext_large_d_320"

clip_backbone = CLIP(model_name=model_name, pretrained=pretrained).to(device_clip)
clip_backbone.clip_model.transformer.batch_first = False
mask_pooling = MaskPooling()

# --------------------------
# Initialize BLIP captioning model on device_blip (cuda:1)
# --------------------------
model_id_or_path = "/home/zhanghaotian/data/blip"
# model_id_or_path = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(model_id_or_path)
blip_model = BlipForConditionalGeneration.from_pretrained(model_id_or_path).to(device_blip)

# --------------------------
# Initialize LDM extractor on cuda:0 (remains on cuda:0)
# --------------------------
extractor = LdmExtractor(model_name="SSD-1B", device=0)

# --- Helper function for semantic association ---
def associate_label(image, mask, clip_backbone, text_classifier, num_templates, mask_pooling, metadata):
    """
    Associate semantic labels to each segment and record prediction logits.
    Returns:
      final_mask: semantic mask with predicted class indices.
      mask_logits: a dictionary mapping each unique mask id to its prediction logit vector 
                   (aligned with noun classes order; note that logits exclude the background class).
    """
    final_mask = torch.zeros_like(mask).to(device_clip)
    mask_logits = {}  # NEW: To store the logits per mask id.
    with torch.no_grad():
        features = clip_backbone(image)
    clip_features = features["clip_vis_dense"]
    
    for i in torch.unique(mask):
        cls_idx = 1.0 * (mask == i.item())
        cls_idx = cls_idx.unsqueeze(-3)
        mask_embed = mask_pooling(clip_features, cls_idx)[0]
        pooled_clip_feature = mask_embed.reshape(1, 1, -1)
        with torch.no_grad():
            pooled_clip_feature = clip_backbone.visual_prediction_forward_convnext(pooled_clip_feature)
        logits = get_classification_logits(pooled_clip_feature, text_classifier, 
                                           clip_backbone.clip_model.logit_scale, num_templates)
        # Save the prediction logits for noun classes (excluding the background logit).
        logits_vector = logits[..., :-1].squeeze().detach().cpu().numpy()
        mask_logits[i.item()] = logits_vector

        probs = logits[..., :-1].softmax(-1)
        idx = torch.argmax(probs).item()
        final_mask[mask == i] = idx
    return final_mask, mask_logits

def mask_refinement(labels, image):
    _, _, h, w = image.shape
    img_size = 1024
    image = F.interpolate(image, size=(img_size, img_size), mode='bilinear')
    masks = torch.cat([1. * (labels == label) for label in torch.unique(labels)], dim=1)
    labels = PAMR(num_iter=30, dilations=[1, 2, 4, 8, 12, 24, 32, 64], device="cuda:1")(image, masks)
    labels = 1. * torch.argmax(labels, dim=1)
    labels = median_filter(labels.cpu().numpy(), 3).astype(int)
    labels = labels[:, None, :, :]
    labels = torch.from_numpy(labels).float()
    labels = F.interpolate(labels, size=(h, w), mode="nearest")
    labels = labels.long()[:, 0, :, :]
    return labels

def process_image(image_path, img_size=1024, t=50, tau=0.5, alpha=10):
    pil_img = Image.open(image_path).convert('RGB')
    # Load image for CLIP and BLIP processing (on cuda:1)
    image = T.ToTensor()(pil_img).unsqueeze(0).to(device_clip)
    _, _, h, w = image.shape

    # --- BLIP Captioning Step ---
    inputs = blip_processor(images=pil_img, return_tensors="pt").to(device_blip)
    caption_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    
    # --- Extract Nouns from the Caption using NLTK ---
    words = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(words)
    nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
    if len(nouns) == 0:
        nouns = ["people"]
    metadata = ['background'] + nouns
    
    # --- Initialize the OV model for this image ---
    ov = open_vocabulary(clip_backbone, metadata, metadata, device="cuda:1")
    text_classifier, num_templates = ov.get_text_classifier()
    
    # --- Semantic Segmentation Step ---
    # Move the image to LDM's device (cuda:0) for processing.
    image_ldm = image.to(torch.device("cuda:0"))
    features = extractor(image_ldm, step=t, img_size=img_size)
    masks = DiffCut().generate_masks(features, tau=tau, mask_size=(h, w), alpha=alpha, img_size=img_size)
    # Move the resulting masks back to device_clip (cuda:1) for further processing.
    masks = torch.Tensor(masks).to(device_clip)
    new_mask = mask_refinement(masks, image).to(device_clip)
    
    # Get semantic mask and prediction logits for each mask id.
    semantic_mask, mask_logits = associate_label(image, new_mask, clip_backbone, text_classifier, num_templates, mask_pooling, metadata)
    semantic_mask = semantic_mask.to(torch.uint8).cpu()

    result = {
        "mask": semantic_mask.numpy(), # (1, 640, 427) # 0,1,2,4
        "caption": caption, # 'a man holding a baseball bat in a park'
        "nouns": metadata, # ['background', 'man', 'baseball', 'bat', 'park']
        "original_mask": new_mask.cpu().numpy(), # (1, 640, 427), 0-8
        "mask_logits": mask_logits, # {0-8: array(5,)}
        "features": features.cpu().numpy() # (1, 1024, 1280)
    }

    return result

# --- HDF5 Saving Functions ---
def load_results_hdf5(hdf5_file):
    existing_keys = set()
    if os.path.exists(hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            existing_keys = set(f.keys())
    return existing_keys

def save_result_to_hdf5(hdf5_file, key, result):
    with h5py.File(hdf5_file, 'a') as f:
        if key in f:
            print(f"Group {key} already exists, skipping.")
        else:
            grp = f.create_group(key)
            grp.create_dataset("mask", data=result["mask"], compression="gzip")
            grp.create_dataset("original_mask", data=result["original_mask"], compression="gzip")  # NEW
            grp.create_dataset("diffusion_feature", data=result["features"], compression="gzip")  # NEW
            # Create a subgroup for mask_logits
            logits_grp = grp.create_group("mask_logits")
            for mask_id, logits in result["mask_logits"].items():
                logits_grp.create_dataset(str(mask_id), data=logits, compression="gzip")
            grp.attrs["caption"] = result["caption"]
            grp.attrs["nouns"] = ','.join(result["nouns"])

def process_all_images(image_dir, hdf5_file, img_size=1024, t=50, tau=0.5, alpha=10):
    # Load existing keys from the HDF5 file (if any)
    existing_keys = load_results_hdf5(hdf5_file)
    if existing_keys:
        print(f"Resuming from {hdf5_file} with {len(existing_keys)} images already processed.")
    
    processed_count = 0
    # List, sort, and filter files based on numeric value in the filename.
    file_list = [filename for filename in os.listdir(image_dir) if filename.endswith(".jpg")]
    file_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # Process only images where the numeric portion is below 20000.
    file_list = [filename for filename in file_list if int(filename.split('_')[-1].split('.')[0]) < 40000]
    
    with h5py.File(hdf5_file, 'a') as f:
        for filename in file_list:
            if filename not in existing_keys:
                image_path = os.path.join(image_dir, filename)
                print(f"Processing {image_path}...")
                result = process_image(image_path, img_size, t, tau, alpha)
                if filename in f:
                    print(f"Group {filename} already exists, skipping.")
                else:
                    grp = f.create_group(filename)
                    grp.create_dataset("mask", data=result["mask"], compression="gzip")
                    grp.create_dataset("original_mask", data=result["original_mask"], compression="gzip")  # NEW
                    grp.create_dataset("diffusion_feature", data=result["features"], compression="gzip")  # NEW
                    # Create a subgroup for mask_logits
                    logits_grp = grp.create_group("mask_logits")
                    for mask_id, logits in result["mask_logits"].items():
                        logits_grp.create_dataset(str(mask_id), data=logits, compression="gzip")
                    grp.attrs["caption"] = result["caption"]
                    grp.attrs["nouns"] = ','.join(result["nouns"])
                processed_count += 1
                # Flush to disk every 200 images.
                if processed_count % 200 == 0:
                    f.flush()
                    print(f"Flushed to disk after processing {processed_count} images.")

                # Clear temporary variables and free up GPU memory.
                del result
                torch.cuda.empty_cache()
                gc.collect()

    print(f"Final results saved with {processed_count} additional images processed.")
    return hdf5_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process images and generate semantic masks using HDF5 for storage.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--hdf5_file", type=str, required=True, help="Path to the output HDF5 file.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist.
    os.makedirs(os.path.dirname(args.hdf5_file), exist_ok=True)

    process_all_images(args.image_dir, args.hdf5_file)