import os
import json
import h5py
import gc
import numpy as np
import nltk
import types


# -- helper to resume --
def load_results_hdf5(hdf5_path):
    if not os.path.exists(hdf5_path):
        return set()
    with h5py.File(hdf5_path, "r") as f:
        return set(f.keys())


gpu = 0
gpu = str(gpu)

# load prompts
with open("train_captions.json") as f:
    train_prompts = json.load(f)
with open("test_captions.json") as f:
    test_prompts = json.load(f)

train_dir = "/home/zhanghaotian/EZ-HOI/hicodet/hico_20160224_det/images/train2015"
test_dir = "/home/zhanghaotian/EZ-HOI/hicodet/hico_20160224_det/images/test2015"

# model init
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
with tf.device("/GPU:" + gpu):
    from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
    from third_party.keras_cv.stable_diffusion import StableDiffusion
    from diffseg.utils import process_image, augmenter
    from diffseg.segmentor import DiffSeg

    image_encoder = ImageEncoder()
    vae = tf.keras.Model(image_encoder.input, image_encoder.layers[-1].output)
    model = StableDiffusion(img_width=512, img_height=512)

# --- MINIMAL PATCH: allow list-of-str prompts in encode_text() ---
_old_encode_text = model.encode_text


def _new_encode_text(self, text):
    if isinstance(text, list):
        # encode each string, then stack
        encs = [_old_encode_text(t) for t in text]
        return tf.stack(encs, axis=0)
    return _old_encode_text(text)


model.encode_text = types.MethodType(_new_encode_text, model)
# --- end patch ---

segmentor = DiffSeg([0.9] * 3, True, 16)
is_noun = lambda pos: pos[:2] == "NN"
is_verb = lambda pos: pos[:1] == "V"

BATCH_SIZE = 4

for mode, image_dir, prompts in [
    # ("train", train_dir, train_prompts),
    # ("train2", train_dir, train_prompts),
    ("test", test_dir, test_prompts)
]:
    # train is first 18000 images train2 is rest images
    if mode == "train2":
        prompts = {
            k: v
            for k, v in prompts.items()
            if int(k.split("_")[-1].split(".")[0]) > 18000
        }
    else:
        prompts = {
            k: v
            for k, v in prompts.items()
            if int(k.split("_")[-1].split(".")[0]) <= 18000
        }
    # prepare file list
    file_list = [
        fn
        for fn in os.listdir(image_dir)
        if fn.lower().endswith((".png", ".jpg", "jpeg"))
    ]
    file_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if mode == "train2":
        file_list = file_list[18000:]
    else:
        file_list = file_list[:18000]
    total_images = len(file_list)
    num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE

    hdf5_file = f"{mode}_attention_maps.h5"
    hdf5_file = os.path.join("/home/zhanghaotian/data129/attention_maps", hdf5_file)
    existing_keys = load_results_hdf5(hdf5_file)
    processed_count = 0

    print(f"[{mode}] Starting: {total_images} images → {num_batches} batches")
    if existing_keys:
        print(
            f"[{mode}]   Resuming from {hdf5_file}, {len(existing_keys)} already done."
        )

    with h5py.File(hdf5_file, "a") as f:
        for batch_idx, i0 in enumerate(range(0, total_images, BATCH_SIZE), start=1):
            batch_fns = file_list[i0 : i0 + BATCH_SIZE]
            print(
                f"[{mode}] Batch {batch_idx}/{num_batches}: preparing {len(batch_fns)} files"
            )
            imgs, prompts_list, names = [], [], []

            # load & preprocess
            for fn in batch_fns:
                key = os.path.splitext(fn)[0]
                if key in existing_keys:
                    continue
                prompt = prompts.get(fn, "")
                prompts_list.append(prompt)
                names.append(fn)

                fp = os.path.join(image_dir, fn)
                with tf.device("/GPU:" + gpu):
                    img, h, w = process_image(fp)
                    imgs.append(augmenter(img))

            if not names:
                print(f"[{mode}]   All {len(batch_fns)} skipped (already done).")
                continue

            imgs_tensor = tf.stack(imgs, axis=0)

            # forward & get attention maps
            with tf.device("/GPU:" + gpu):
                latents = vae(imgs_tensor, training=False)
                _, w64, w32, w16, w8, x64, x32, x16, x8 = model.text_to_image(
                    prompts_list,
                    batch_size=len(names),
                    latent=latents,
                    timestep=300,
                )

            # save per-image
            for idx, fn in enumerate(names):
                key = os.path.splitext(fn)[0]
                prompt = prompts_list[idx]
                tokens = nltk.word_tokenize(prompt)
                tags = nltk.pos_tag(tokens)
                n_l = [(i, w) for i, (w, p) in enumerate(tags) if is_noun(p)]
                v_l = [(i, w) for i, (w, p) in enumerate(tags) if is_noun(p)]
                xw = segmentor.aggregate_x_weights(
                    [x64[idx], x32[idx], x16[idx], x8[idx]], weight_ratio=[1.0] * 4
                )[0]

                if v_l:
                    ids = [i + 1 for i, _ in v_l]
                    res = {"mean": np.mean(xw[:, :, ids], axis=-1)}
                elif n_l:
                    ids = [i + 1 for i, _ in n_l]
                    res = {"mean": np.mean(xw[:, :, ids], axis=-1)}
                else:
                    all_ids = list(range(1, len(tokens) + 1))
                    res = {"mean": np.mean(xw[:, :, all_ids], axis=-1)}

                grp = f.create_group(key)
                for attn_key, attn in res.items():
                    grp.create_dataset(
                        attn_key, data=attn.astype("float16"), compression="gzip"
                    )

                processed_count += 1
                print(f"[{mode}]   Saved '{fn}' ({processed_count} total new)")

                # clear memory per image
                tf.keras.backend.clear_session()
                gc.collect()

            # flush every batch
            f.flush()
            print(f"[{mode}]   Flushed HDF5 after batch {batch_idx}")

    print(f"[{mode.capitalize()} done] +{processed_count} new images → {hdf5_file}\n")

print("All modes complete.")
