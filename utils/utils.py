import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_img_and_mask(img, mask):
    if isinstance(img, torch.Tensor):
        img = img.squeeze().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()

    classes = int(mask.max()) + 1
    fig, ax = plt.subplots(1, classes + 1, figsize=(12, 4))

    ax[0].set_title('Input image')
    ax[0].imshow(img, cmap='gray')
    ax[0].axis('off')

    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i})')
        ax[i + 1].imshow(mask == i, cmap='gray')
        ax[i + 1].axis('off')

    plt.tight_layout()
    plt.show()