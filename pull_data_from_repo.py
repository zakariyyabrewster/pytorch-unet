import requests
import os
from PIL import Image
from pathlib import Path
import random
import yaml

class PullDataFromRepo:
    def __init__(self, config):
        self.config = config
        self.repo_owner = self.config['pull_dataset']['repo_owner']
        self.repo_name = self.config['pull_dataset']['repo_name']
        self.img_folder_path = self.config['pull_dataset']['img_folder_path']
        self.mask_folder_path = self.config['pull_dataset']['mask_folder_path']
        self.branch = self.config['pull_dataset'].get('branch', 'main')
        self.base_api_url = os.path.join(self.config['pull_dataset']['base_api_url'], self.repo_owner, self.repo_name, "contents/")


    def fetch_filenames(self):
        # Images
        response = requests.get(self.base_api_url + self.img_folder_path)
        if response.status_code == 200:
            files = response.json()
            image_urls = [file["download_url"] for file in files if file["name"].endswith((".png"))]

        else:
            print("failed to fetch image file list.")

        # Masks
        response = requests.get(self.base_api_url + self.mask_folder_path)
        if response.status_code == 200:
            files = response.json()
            masks_urls = [file["download_url"] for file in files if file["name"].endswith((".png"))]
        else:
            print("Failed to fetch mask file list.")


        # Pull file names
        image_names = {url.split("/")[-1] for url in image_urls}
        mask_names = {url.split("/")[-1] for url in masks_urls}


        # Process mask names by removing the "__pixels0" suffix
        processed_mask_names = {name.replace(self.config['pull_dataset']['mask_suffix'], "") for name in mask_names}

        # Check which images have masks
        self.image_names = processed_mask_names.intersection(image_names)
        self.mask_names = {name for name in mask_names if name.replace(self.config['pull_dataset']['mask_suffix'], "") in image_names}

        # Number of images
        print(f'Number of Images: {len(image_names)}')

# Create a folder in Colab (or Google Drive if mounted)
    def create_folders(self):

        img_dest = os.path.join('datasets', self.config['dataset']['data_name'], self.config['dataset']['img_dir'])
        mask_dest = os.path.join('datasets', self.config['dataset']['data_name'], self.config['dataset']['mask_dir'])
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(mask_dest, exist_ok=True)
        valid_image_urls = sorted([f"https://raw.githubusercontent.com/{self.repo_owner}/{self.repo_name}/{self.branch}/{self.img_folder_path}/"+name for name in self.image_names])
        valid_masks_urls = sorted([f"https://raw.githubusercontent.com/{self.repo_owner}/{self.repo_name}/{self.branch}/{self.mask_folder_path}/"+name for name in self.mask_names])

# Download images
        for url in valid_image_urls:
            filename = os.path.join(img_dest, url.split("/")[-1])
            response = requests.get(url)

            if response.status_code == 200:
                with open(filename, "wb") as file:
                    file.write(response.content)
            else:
                print(f"Failed to download: {url}")

        for url in valid_masks_urls:
            filename = os.path.join(mask_dest, url.split("/")[-1])
            response = requests.get(url)

            if response.status_code == 200:
                with open(filename, "wb") as file:
                    file.write(response.content)
            else:
                print(f"Failed to download: {url}")