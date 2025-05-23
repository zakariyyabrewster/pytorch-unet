import requests
import os
from PIL import Image
from pathlib import Path
import random

repo_owner = "YimianDai"
repo_name = "open-sirst-v2"
folder_paths = ["images/targets", "annotations/masks"]
base_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"

# Images
response = requests.get(base_api_url+folder_paths[0])
if response.status_code == 200:
    files = response.json()
    image_urls = [file["download_url"] for file in files if file["name"].endswith((".png")) and file["name"].startswith(("Misc_"))]

else:
  print("failed to fetch file list.")

# Masks
response = requests.get(base_api_url+folder_paths[1])
if response.status_code == 200:
    files = response.json()
    masks_urls = [file["download_url"] for file in files if file["name"].endswith((".png")) and file["name"].startswith(("Misc_"))]

else:
    print("Failed to fetch file list.")

# Pull file names
image_names = {url.split("/")[-1] for url in image_urls}
mask_names = {url.split("/")[-1] for url in masks_urls}


# Process mask names by removing the "__pixels0" suffix
processed_mask_names = {name.replace("_pixels0", "") for name in mask_names}

# Check which images have masks
image_names = processed_mask_names.intersection(image_names)

# Number of images
print(len(image_names))

# Create a folder in Colab (or Google Drive if mounted)

valid_image_urls = sorted(["https://raw.githubusercontent.com/YimianDai/open-sirst-v2/master/images/targets/"+name for name in image_names])
valid_masks_urls = sorted(["https://raw.githubusercontent.com/YimianDai/open-sirst-v2/master/annotations/masks/"+name for name in mask_names])

# Download images
for url in valid_image_urls:
    filename = os.path.join("data/imgs", url.split("/")[-1])
    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download: {url}")

for url in valid_masks_urls:
    filename = os.path.join("data/masks", url.split("/")[-1])
    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download: {url}")

import os
from PIL import Image
import random

def get_image_mask_pairs(images_dir, masks_dir, valid_exts={'.png', '.jpg', '.jpeg'}):
    pairs = []
    for filename in os.listdir(images_dir):
        base, ext = os.path.splitext(filename)
        if ext.lower() in valid_exts:
            image_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, base + '_pixels0.png')
            if os.path.exists(mask_path):
                pairs.append((image_path, mask_path))
            else:
                print(f"No mask found for {filename}")
    return pairs

def resize_or_pad_image_and_mask(img: Image.Image, mask: Image.Image, size: int, seed=None):
    if seed is not None:
        random.seed(seed)

    img = img.convert("L")
    mask = mask.convert("L")
    w, h = img.size

    if w >= size and h >= size:
        left = random.randint(0, w - size)
        top = random.randint(0, h - size)
        return img.crop((left, top, left + size, top + size)), mask.crop((left, top, left + size, top + size))

    elif w < size and h < size:
        bg_img = Image.new("L", (size, size), 0)
        bg_mask = Image.new("L", (size, size), 0)
        offset_x = random.randint(0, size - w)
        offset_y = random.randint(0, size - h)
        bg_img.paste(img, (offset_x, offset_y))
        bg_mask.paste(mask, (offset_x, offset_y))
        return bg_img, bg_mask

    else:
        scale = size / w if w < size else size / h
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        if new_w >= size and new_h >= size:
            left = random.randint(0, new_w - size)
            top = random.randint(0, new_h - size)
            return img.crop((left, top, left + size, top + size)), mask.crop((left, top, left + size, top + size))
        else:
            bg_img = Image.new("L", (size, size), 0)
            bg_mask = Image.new("L", (size, size), 0)
            offset_x = random.randint(0, size - new_w)
            offset_y = random.randint(0, size - new_h)
            bg_img.paste(img, (offset_x, offset_y))
            bg_mask.paste(mask, (offset_x, offset_y))
            return bg_img, bg_mask

# Process and overwrite
image_mask_pairs = get_image_mask_pairs("data/imgs", "data/masks")

for img_path, mask_path in image_mask_pairs:
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    processed_img, processed_mask = resize_or_pad_image_and_mask(img, mask, size=480)
    
    processed_img.save(img_path)
    processed_mask.save(mask_path)