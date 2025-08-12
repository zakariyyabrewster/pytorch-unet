# Pytorch Implementation of U-Net
Based on Model ideated by Olaf Ronneberger, *"U-Net: Convolutional Networks for Biomedical Image Segmentation"* and implementation by GitHub user [@milesial](https://github.com/milesial) [https://github.com/milesial/Pytorch-UNet]

This repo includes:
- Model Architecture
- Sample dataset taken from [https://github.com/YimianDai/open-sirst-v2]: Resized and Normalized to fit into model
- Training Pipeline, including:
  - Training Loop
  - Validation Mechanism
  - Testing Loop
  - Evaluation on extended metrics (Precision-Recall, ROC) at various thresholds
- Configuration File
  - Training Pipeline Params (epochs, batch size)
  - Optimizer Params (learning rate, weight decay, momentum)
  - Pathing Details (directory names)
  - Repository Details for pulling data
- Repository Access Script
  - Access images and masks directly from repo, generate folders for dataset
- Demo Notebooks (visualization of graphs and logging of training process, progress and results)


## Setup
### Initializing Images and Masks
#### From Own Computer
In the datasets folder, create a directory titled after the name of the dataset (Ex. SIRST_V2). Into this folder, move your image and mask folders. Each image should have a corresponding mask of the same name + a common suffix across the dataset (Ex. for SIRST_V2, mask_suffix = "_pixels0"). Ensure that the name of the folder holding the images and masks matches ```data_name``` in the configuration file. Additionally, match the name of both the image and mask directories to ```img_dir``` and ```mask_dir``` in the configuration file. Note the mask suffix and log it in the configuratio file under ```mask_suffix```. 

### From GitHub Repo
Note the ```pull_dataset``` section of the configuration file. Find the repository you wish to pull from, and copy the repo name, repo owner, branch, path to the images and corresponding masks you wish to use, and the mask suffix into the configuration file.
If you wish to use custom naming for the dataset, update the configuration ```dataset``` section as you wish. If you want to use the repository-given name, leave ```data_name``` as ```None```. ```mask_suffix``` will also carry over from the ```pull_data``` configuration if it is ```None``` in the ```dataset``` section.

### Environment
#### Terminal
```
git clone https://github.com/zakariyyabrewster/pytorch-unet
cd pytorch-unet # or relative pathing to repo
```
If you want to setup a virtual environment:
```
virtualenv myenv
```
pip install -r requirements.txt

```


