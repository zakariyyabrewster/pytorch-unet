import os
import numpy as np
from PIL import Image
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

def load_mask(path):
    return np.array(Image.open(path)) // 255

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

        recall_array = np.array(recall_list)
        precision_array = np.array(precision_list)
        sort_idx = np.argsort(recall_array)
        recall_array = recall_array[sort_idx]
        precision_array = precision_array[sort_idx]

        # Compute area under the curve using trapezoidal integration
        AP = np.trapz(precision_array, recall_array)
        
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        print(f"Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}, AP: {AP:.4f}")
        plt.figure()
        plt.scatter(recall_list, precision_list, marker='.', label='Precision-Recall curve')
        plt.scatter(mean_recall, mean_precision, marker='o', color='red', s=100, label=f'Mean Precision-Recall: P={mean_precision:.4f}, R={mean_recall:.4f}')
        plt.axhline(y=AP, color='blue', linestyle='--', label=f'AP = {AP:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig('metrics/PR_curve.png')

    def ROC(self):
        fpr_list = []
        tpr_list = []
        for metric in self.metrics_list:
            fpr_list.append(metric.fpr)
            tpr_list.append(metric.tpr)
        
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        sort_idx = np.argsort(fpr_array)
        fpr_array = fpr_array[sort_idx]
        tpr_array = tpr_array[sort_idx]

        # Compute area under the curve using trapezoidal integration
        AUC = np.trapz(tpr_array, fpr_array)

        mean_fpr = np.mean(fpr_list)
        mean_tpr = np.mean(tpr_list)
        print(f"Mean FPR: {mean_fpr:.4f}, Mean TPR: {mean_tpr:.4f}, AUC: {AUC:.4f}")
        plt.figure()
        plt.scatter(fpr_list, tpr_list, marker='.', label='ROC curve')
        plt.scatter(mean_fpr, mean_tpr, marker='o', color='red', s=100, label=f'Mean ROC: TPR={mean_tpr:.4f}, FPR={mean_fpr:.4f}')
        plt.axhline(y=AUC, color='blue', linestyle='--', label=f'AUC = {AUC:.4f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('metrics/ROC_curve.png')
    
    def display_metrics(self):
        print(f"Mean IoU: {self.mIoU:.4f}")
        print(f"Mean F-measure: {self.mFmeasure:.4f}")
        self.PrecisionRecall()
        self.ROC()

if __name__ == '__main__':
    model_metrics = []
    os.makedirs('metrics', exist_ok=True)
    gt_dir = 'data/masks'
    pred_dir = 'data/outputs'
    gt_files = sorted(glob(os.path.join(gt_dir, '*.png')))
    for gt_path in gt_files:
        base_name = os.path.splitext(os.path.basename(gt_path))[0]
        pred_name = base_name + '_OUT.png'
        pred_path = os.path.join(pred_dir, pred_name)
        if not os.path.exists(pred_path):
            print(f"Prediction file {pred_path} does not exist.")
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)
        assert gt.shape == pred.shape, f"Shape mismatch: {gt.shape} vs {pred.shape}"
        metrics_i = EvalMetrics(gt, pred)
        model_metrics.append(metrics_i)
    
    eval_metrics = EvalMetrics(model_metrics)
    eval_metrics.display_metrics()
