import argparse
import os
import random
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import logging
from utils.dice_score import dice_loss
from eval_metrics import *
from pull_data_from_repo import *
import pandas as pd


from unet import UNet
from utils.data_loading import BasicDataset


class Pipeline(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = os.path.join(self.config['log_dir'], self.config['dataset']['data_name'])
        self.graph_dir = os.path.join(self.log_dir, 'graphs')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()

        self.criterion = nn.BCEWithLogitsLoss()

        self.config['dataset']['img_dir'] = os.path.join('datasets', 
                                                         self.config['dataset']['data_name'],
                                                         self.config['dataset']['img_dir'])
        self.config['dataset']['mask_dir'] = os.path.join('datasets', 
                                                         self.config['dataset']['data_name'],
                                                         self.config['dataset']['mask_dir'])
        dataset_config = self.config['dataset'].copy()
        dataset_config.pop('data_name', None)  # Remove data_name from dataset config

        self.dataset = BasicDataset(**dataset_config)
        n_val = int(len(self.dataset) * self.config['dataloader']['val_ratio'])
        n_test = int(len(self.dataset) * self.config['dataloader']['test_ratio'])
        n_train = len(self.dataset) - n_val - n_test
        train_set, val_set, test_set = random_split(
            self.dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.config['random_seed'])
        )
        self.train_loader = DataLoader(train_set, batch_size=self.config['batch_size'], shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_set, batch_size=self.config['batch_size'], shuffle=False, pin_memory=True)
        self.test_loader = DataLoader(test_set, batch_size=self.config['batch_size'], shuffle=False, pin_memory=True)

        self.eval_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True) if self.config['eval_metrics']['eval_on_test'] else DataLoader(self.dataset, batch_size=1, shuffle=False, pin_memory=True)

        self.train_losses = []
        self.train_epochs = list(range(1, config['epochs'] + 1))
        self.val_losses = []
        self.val_epochs = []
        self.log(f"Pipeline initialized for {self.config['dataset']['data_name']} Dataset")
        self.log(f'All files will be saved to {self.log_dir}')
        self.log(f'Train set size: {len(train_set)}\nValidation set size: {len(val_set)}\nTest set size: {len(test_set)}')
        self.log("===========================================================")

    def _setup_logging(self):
        """Setup logging to both console and file"""
        # Create log file path
        log_file = os.path.join(self.log_dir, 'pipeline_log.txt')
        
        # Setup logger
        self.logger = logging.getLogger('Pipeline')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def log(self, message):
        """Custom logging function that logs to both console and file"""
        self.logger.info(message)

    def _train(self):
        best_val_loss = np.inf
        model = UNet(**self.config['model']).to(self.device)
        optim_config = self.config['optim'].copy()
        optim_config.pop('optimizer', None)  # Remove optimizer from config if it exists
        optimizer = optim.RMSprop(model.parameters(), **optim_config, foreach=True)
        self.log(f'Training Model on {self.config["dataset"]["data_name"]}...' )
        self.log("===========================================================")

        # Training loop
        for epoch in range(1, self.config['epochs'] + 1):
            model.train()
            epoch_loss = 0.0
            for batch in self.train_loader:
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                assert images.shape[1] == model.n_channels, f'Expected {model.n_channels} channels, got {images.shape[1]}'
                # Add channel dimension to masks if needed for loss computation
                if masks.ndim == 3:  # (batch, height, width)
                    masks = masks.unsqueeze(1)  # (batch, 1, height, width)
                assert images.shape[2:] == masks.shape[2:], f'Spatial dimensions must match: {images.shape[2:]} vs {masks.shape[2:]}'
                optimizer.zero_grad()
                preds = model(images)
                loss = self.criterion(preds, masks) + dice_loss(preds, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            if epoch % self.config['log_every_n_epochs'] == 0:
                self.log(f'Epoch {epoch}/{self.config["epochs"]}, Average Loss Per Sample: {avg_epoch_loss:.4f}')
            self.train_losses.append(avg_epoch_loss)

            if epoch % self.config['eval_every_n_epochs'] == 0:
                val_loss = self._validate(model, epoch)
                self.val_losses.append(val_loss)
                self.val_epochs.append(epoch)
                self.log(f'Validation loss at Epoch {epoch}/{self.config["epochs"]}: {val_loss:.4f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(self.log_dir, 'final_model.pth'))
                    self.log(f'New best model saved to {os.path.join(self.log_dir, "final_model.pth")} with validation loss: {val_loss:.4f}')

        self._plot_losses(train_losses=self.train_losses, val_losses=self.val_losses,
                          train_epochs=self.train_epochs, val_epochs=self.val_epochs, graph_dir=self.graph_dir)
        self.log(f'Training completed. Best validation loss: {best_val_loss:.4f}')

        torch.save(model.state_dict(), os.path.join(self.log_dir, 'final_model.pth'))

        self.model = model
        self.log(f'Model saved to {os.path.join(self.log_dir, "final_model.pth")}')
            
    
    def _validate(self, model, epoch):
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                # Add channel dimension to masks if needed
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                preds = model(images)
                loss = self.criterion(preds, masks) + dice_loss(preds, masks)
                total_loss += loss.item()
                n_batches += 1
        val_loss = total_loss / n_batches
        self.log(f'Validation loss at Epoch {epoch}: {val_loss:.4f}')

        return val_loss
    
    def _test(self):
        model_path = os.path.join(self.log_dir, 'final_model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        total_loss = 0.0
        n_batches = 0
        total_dice = 0.0
        self.log(f'Testing Model on {self.config["dataset"]["data_name"]}...')

        with torch.no_grad():
            self.model.eval()
            for batch in self.test_loader:
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                # Add channel dimension to masks if needed
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                preds = self.model(images)
                loss = self.criterion(preds, masks) + dice_loss(preds, masks)
                dice = (1 - dice_loss(preds, masks)).item()
                total_dice += dice
                total_loss += loss.item()
                n_batches += 1

        test_loss = total_loss / n_batches if n_batches > 0 else 0.0
        mean_dice = total_dice / n_batches if n_batches > 0 else 0.0
        self.log(f'Test loss: {test_loss:.4f}, Mean Dice: {mean_dice:.4f}')

    def _eval_metrics(self):
        model_path = os.path.join(self.log_dir, 'final_model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        thresholds = np.linspace(0, 1, self.config['eval_metrics']['n_thresholds'])
        
        thresh_list = []
        precisions = []
        recalls = []
        tprs = []
        fprs = []
        self.log(f'Evaluating metrics...')
        self.log("===========================================================")
        for t in thresholds:
            self.log(f'Threshold: {t:.2f}')
            all_metrics = []
            for batch in self.eval_loader:
                image, mask = batch['image'].to(self.device), batch['mask'].to(self.device)
                gt = mask.cpu().numpy().astype(np.uint8)
                pred = self._predict(model=self.model, img=image, out_threshold=t, device=self.device)
                metrics_i = InstanceMetrics(gt.squeeze(0, 1), pred.squeeze(0, 1))
                all_metrics.append(metrics_i)
            all_metrics_eval = EvalMetrics(all_metrics)
            if self.config['eval_metrics']['plot_curves_per_threshold']:
                all_metrics_eval.plot_curves()
            thresh_list.append(t)
            precisions.append(all_metrics_eval.meanPR[1])
            recalls.append(all_metrics_eval.meanPR[0])
            tprs.append(all_metrics_eval.meanROC[0])
            fprs.append(all_metrics_eval.meanROC[1])
        
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(recalls, precisions, marker='o')
        for x, y, t in zip(recalls, precisions, thresholds):
            ax.annotate(f'{t:.2f}', (x, y), textcoords="offset points", xytext=(5,-5))
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Mean Precision–Recall at Different Thresholds')
        ax.grid(True)

        fig.savefig(os.path.join(self.graph_dir, 'meanPR_by_threshold.png'))
        self.log(f'Mean Precision-Recall plot saved to {os.path.join(self.graph_dir, "meanPR_by_threshold.png")}')

        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(fprs, tprs, marker='o')
        for x, y, t in zip(fprs, tprs, thresholds):
            ax.annotate(f'{t:.2f}', (x, y), textcoords="offset points", xytext=(5,-5))
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Mean ROC at Different Thresholds')
        ax.grid(True)
        fig.savefig(os.path.join(self.graph_dir, 'meanROC_by_threshold.png'))
        self.log(f'Mean ROC plot saved to {os.path.join(self.graph_dir, "meanROC_by_threshold.png")}')

        if self.config['eval_metrics']['viz']:
            plt.show()



    def _predict(self, model, img, out_threshold, device):
        model.eval()
        with torch.no_grad():
            pred = model(img)
            pred = torch.sigmoid(pred) > out_threshold # (1, 1, H, W)
        
        return pred.cpu().numpy().astype(np.uint8)
    

    def _plot_losses(self, train_losses, val_losses, train_epochs, val_epochs, graph_dir):
        plt.figure(figsize=(10, 5))
        plt.plot(train_epochs, train_losses, label='Training Loss')
        plt.plot(val_epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(graph_dir, 'train_val_loss_plot.png'))
        plt.close()
        self.log(f'Loss plot saved to {os.path.join(graph_dir, "train_val_loss_plot.png")}')
        if self.config['eval_metrics']['viz']:
            plt.show()



if __name__ == '__main__':
    config = yaml.load(open('config_mdm.yaml', "r"), Loader=yaml.FullLoader)

    if config['pull_dataset']['enabled']:
        if config['dataset']['data_name'] is None:
            config['dataset']['data_name'] = config['pull_dataset']['repo_name']
        if config['dataset']['mask_suffix'] is None:
            config['dataset']['mask_suffix'] = config['pull_dataset']['mask_suffix']

        pull_data = PullDataFromRepo(config)
        pull_data.fetch_filenames()
        pull_data.create_folders()

    pipeline = Pipeline(config)
    
    # Log the configuration
    pipeline.log("Configuration:")
    pipeline.log(str(config))
    pipeline.log("===========================================================")
    
    # Run pipeline
    pipeline._train()
    pipeline._test()
    pipeline._eval_metrics()

    pipeline.log("Pipeline completed successfully.")



