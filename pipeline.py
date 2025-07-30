import argparse
import logging
import os
import random
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils.dice_score import dice_loss
from eval_metrics import *


from unet import UNet
from utils.data_loading import BasicDataset


def Pipeline(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = self.config['log_dir']
        self.criterion = nn.BCEWithLogitsLoss()
        self.dataset = BasicDataset(**config['dataset'])
        n_val = int(len(self.dataset) * config['dataloader']['val_ratio'])
        n_test = int(len(self.dataset) * config['dataloader']['test_ratio'])
        n_train = len(self.dataset) - n_val - n_test
        train_set, val_set, test_set = random_split(
            self.dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.config['random_seed'])
        )
        self.train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
        self.test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

        self.train_losses = []
        self.train_epochs = list(range(1, config['epochs'] + 1))
        self.val_losses = []
        self.val_epochs = []

        print(f'Train set size: {len(train_set)}\nValidation set size: {len(val_set)}\nTest set size: {len(test_set)}')

    def _train(self):
        best_val_loss = np.inf
        model = UNet(**config['model']).to(self.device)
        optimizer = optim.RMSprop(model.parameters(), **config['optim'], foreach=True)
        global_step = 0
        for epoch in range(1, self.config['epochs'] + 1):
            model.train()
            epoch_loss = 0.0
            for batch in self.train_loader:
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                assert images.shape[1] == model.n_channels, f'Expected {model.n_channels} channels, got {images.shape[1]}'
                assert images.shape == masks.shape
                optimizer.zero_grad()
                preds = model(images)
                loss = self.criterion(preds, masks) + dice_loss(preds, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f'Epoch {epoch}/{self.config["epochs"]}, Average Loss Per Sample: {avg_epoch_loss:.4f}')
            self.train_losses.append(avg_epoch_loss)

            if epoch % self.config['eval_every_n_epochs'] == 0:
                val_loss = self._validate(model, epoch)
                self.val_losses.append(val_loss)
                self.val_epochs.append(epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(self.log_dir, 'final_model.pth'))
            
        self._plot_losses(train_losses=self.train_losses, val_losses=self.val_losses, 
                          train_epochs=self.train_epochs, val_epochs=self.val_epochs, log_dir=self.log_dir)
        self.model = model
            
    
    def _validate(self, model, epoch):
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                preds = model(images)
                loss = self.criterion(preds, masks) + dice_loss(preds, masks)
                total_loss += loss.item()
                n_batches += 1
        val_loss = total_loss / n_batches
        print(f'Validation loss at Epoch {epoch}: {val_loss:.4f}')

        return val_loss
    
    def _test(self):
        model_path = os.path.join(self.log_dir, 'final_model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            self.model.eval()
            for batch in self.test_loader:
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                preds = self.model(images)
                loss = self.criterion(preds, masks) + dice_loss(preds, masks)
                total_loss += loss.item()
                n_batches += 1

        test_loss = total_loss / n_batches if n_batches > 0 else 0.0
        print(f'Test loss: {test_loss:.4f}')

    def _eval_metrics(self):
        model_path = os.path.join(self.log_dir, 'final_model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        

        metrics_list = []
        with torch.no_grad():
            for batch in self.test_loader:
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                preds = self.model(images)
                preds = torch.sigmoid(preds) > 0.5

    def _plot_losses(self, train_losses, val_losses, train_epochs, val_epochs, log_dir):
        plt.figure(figsize=(10, 5))
        plt.plot(train_epochs, train_losses, label='Training Loss')
        plt.plot(val_epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(log_dir, 'loss_plot.png'))
        plt.close()
        print(f'Loss plot saved to {os.path.join(log_dir, "loss_plot.png")}')







if __name__ == '__main__':
    config = yaml.load(open('config_unet.yaml', "r"), Loader=yaml.FullLoader)
    print(config)
    os.makedirs(config['log_dir'], exist_ok=True)

    pipeline = Pipeline(config)
    pipeline.train()

