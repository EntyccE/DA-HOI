import os, json
# point at your mirror and single GPU
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

from PIL import Image
import nltk
from transformers import AutoProcessor, TFBlipForConditionalGeneration

image_dir = "/home/zhanghaotian/diffseg/test_images/"
model_name = "/home/zhanghaotian/data/blip"

# load BLIP once
processor = AutoProcessor.from_pretrained(model_name)
blip = TFBlipForConditionalGeneration.from_pretrained(model_name)

def is_image(fp):
    try:
        with Image.open(fp) as img:
            img.verify()
        return True
    except:
        return False

prompts = {}
for i, fn in enumerate(os.listdir(image_dir)):
    if i == 20:
        break
    fp = os.path.join(image_dir, fn)
    if not is_image(fp):
        continue
    name = os.path.splitext(fn)[0]
    inputs = processor(images=Image.open(fp), return_tensors="tf")
    out = blip.generate(**inputs)
    prompt = processor.decode(out[0], skip_special_tokens=True)
    prompts[name] = prompt
    print(f"{name} → {prompt}")
    

# save all captions
with open("prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(">> captions written to prompts.json")