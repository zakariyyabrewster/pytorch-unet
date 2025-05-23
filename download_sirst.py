import requests
import os
from PIL import Image
from pathlib import Path

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

def resize_folder_images(folder_path, size=(480, 480)):
    folder = Path(folder_path)
    for file in folder.iterdir():
        if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            img = Image.open(file)
            img = img.resize(size, resample=Image.NEAREST if 'mask' in str(folder) else Image.BICUBIC)
            img.save(file)

# Resize images
resize_folder_images('data/imgs', size=(480, 480))

# Resize masks (use NEAREST interpolation to preserve class labels)
resize_folder_images('data/masks', size=(480, 480))


