import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt

def load_mask(path):
    return np.array(Image.open(path)) // 255

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    dummy = mask[0].cpu().squeeze().numpy()
    return dummy

class InstanceMetrics:
    def __init__(self, gt, pred):
        self.gt = gt.astype(bool)
        self.pred = pred.astype(bool)
        self.tp = np.sum(self.gt & self.pred)
        self.tn = np.sum(~self.gt & ~self.pred)
        self.fp = np.sum(~self.gt & self.pred)
        self.fn = np.sum(self.gt & ~self.pred)
        self.union = np.sum(self.gt) + np.sum(self.pred) - self.tp
        self.beta = 0.3
        self.precision = self.tp / (self.tp + self.fp + 1e-8)
        self.recall = self.tp / (self.tp + self.fn + 1e-8)
        self.f_measure = ((1 + self.beta ** 2) * self.precision * self.recall) / (self.beta ** 2 * self.precision + self.recall + 1e-8)
        self.tpr = self.recall
        self.fpr = self.fp / (self.fp + self.tn + 1e-8)
        self.iou = self.tp / (self.union + 1e-8)
    
    def __str__(self):
        return f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F-measure: {self.f_measure:.4f} \nTrue Positive Rate (TPR): {self.tpr:.4f}, False Positive Rate (FPR): {self.fpr:.4f}, IoU: {self.iou:.4f}"

class EvalMetrics:
    def __init__(self, metrics_list):
        self.metrics_list = metrics_list
        self.mIoU = np.mean([metric.iou for metric in metrics_list])
        self.mFmeasure = np.mean([metric.f_measure for metric in metrics_list])
    
    def PrecisionRecall(self):
        precision_list = []
        recall_list = []
        for metric in self.metrics_list:
            precision_list.append(metric.precision)
            recall_list.append(metric.recall)

        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        return mean_precision, mean_recall

    def ROC(self):
        fpr_list = []
        tpr_list = []
        for metric in self.metrics_list:
            fpr_list.append(metric.fpr)
            tpr_list.append(metric.tpr)

        mean_fpr = np.mean(fpr_list)
        mean_tpr = np.mean(tpr_list)

        return mean_fpr, mean_tpr
    
    def display_metrics(self):
        print(f"Mean IoU: {self.mIoU:.4f}")
        print(f"Mean F-measure: {self.mFmeasure:.4f}")
        self.PrecisionRecall()
        self.ROC()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
    #                     help='Minimum probability value to consider a mask pixel white')
    # parser.add_argument('--scale', '-s', type=float, default=0.5,
    #                     help='Scale factor for the input images')
    # parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting prediction pipeline')
    args = get_args()
    img_dir = "data/imgs"
    gt_dir = "data/masks"
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    mask_threshold = np.linspace(0, 1, 11)

    # bin_masks_dir = 'data/preds'
    # os.makedirs(bin_masks_dir, exist_ok=True)
    in_files = glob.glob(os.path.join(img_dir, '*.png'))
    model_metrics = []
    for thresh in mask_threshold:
        model_metrics_i = []
        for img_path in in_files:
            img = Image.open(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            gt_name = base_name + 'pixels0.png'
            gt_path = os.path.join(gt_dir, gt_name)
            gt = load_mask(gt_path)
            pred = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=thresh,
                            device=device) # pred is a numpy array
            assert gt.shape == pred.shape, f"Shape mismatch: {gt.shape} vs {pred.shape}"
            metrics_i = InstanceMetrics(gt, pred)
            model_metrics_i.append(metrics_i)
        eval_metrics = EvalMetrics(model_metrics_i)
        logging.info(f"Metrics for threshold {thresh}: {eval_metrics}")
        model_metrics.append(eval_metrics)
    
    plt.figure()
    for i, metrics in enumerate(model_metrics):
        mean_precision, mean_recall = metrics.PrecisionRecall()
        plt.scatter(mean_recall, mean_precision, marker='o',label=f'Threshold {mask_threshold[i]:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('metrics/PR_curve.png')

    plt.figure()
    for i, metrics in enumerate(model_metrics):
        mean_fpr, mean_tpr = metrics.ROC()
        plt.scatter(mean_fpr, mean_tpr, marker='o', label=f'Threshold {mask_threshold[i]:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('metrics/ROC_curve.png')

    print("Mean IoU and F-measure for each threshold:")
    for i, metrics in enumerate(model_metrics):
        print(f"Threshold {mask_threshold[i]:.2f} - Mean IoU: {metrics.mIoU:.4f}, Mean F-measure: {metrics.mFmeasure:.4f}")
        

        







