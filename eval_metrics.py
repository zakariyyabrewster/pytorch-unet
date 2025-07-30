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
        self._computePR()
        self._computeROC()
        os.makedirs('metrics', exist_ok=True)

    def _computePR(self):
        self.precision_list = []
        self.recall_list = []
        for metric in self.metrics_list:
            self.precision_list.append(metric.precision)
            self.recall_list.append(metric.recall)

        recall_array = np.array(self.recall_list)
        precision_array = np.array(self.precision_list)
        sort_idx = np.argsort(recall_array)
        recall_array = recall_array[sort_idx]
        precision_array = precision_array[sort_idx]

        # Compute area under the curve using trapezoidal integration
        self.AP = np.trapz(precision_array, recall_array)

        mean_precision = np.mean(self.precision_list)
        mean_recall = np.mean(self.recall_list)
        self.meanPR = (mean_recall, mean_precision)
        print(f"Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}, AP: {self.AP:.4f}")
    
    def plot_curves(self):
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 6))

        # — ROC subplot —
        ax_roc.scatter(self.fpr_list, self.tpr_list, marker='.', label='ROC curve')
        ax_roc.scatter(
            self.meanROC[0], self.meanROC[1],
            marker='o', color='red', s=100,
            label=f'Mean ROC: TPR={self.meanROC[1]:.4f}, FPR={self.meanROC[0]:.4f}'
        )
        ax_roc.axhline(
            y=self.AUC, color='blue', linestyle='--',
            label=f'AUC = {self.AUC:.4f}'
        )
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        ax_roc.grid(True)

        # — Precision–Recall subplot —
        ax_pr.scatter(self.recall_list, self.precision_list, marker='.', label='PR curve')
        ax_pr.scatter(
            self.meanPR[1], self.meanPR[0],
            marker='o', color='red', s=100,
            label=f'Mean PR: P={self.meanPR[0]:.4f}, R={self.meanPR[1]:.4f}'
        )
        ax_pr.axhline(
            y=self.AP, color='blue', linestyle='--',
            label=f'AP = {self.AP:.4f}'
        )
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision–Recall Curve')
        ax_pr.legend()
        ax_pr.grid(True)

        plt.tight_layout()
        fig.savefig('metrics/combined_ROC_PR.png')
        plt.show()
        

    def _computeROC(self):
        self.fpr_list = []
        self.tpr_list = []
        for metric in self.metrics_list:
            self.fpr_list.append(metric.fpr)
            self.tpr_list.append(metric.tpr)

        fpr_array = np.array(self.fpr_list)
        tpr_array = np.array(self.tpr_list)
        sort_idx = np.argsort(fpr_array)
        fpr_array = fpr_array[sort_idx]
        tpr_array = tpr_array[sort_idx]

        # Compute area under the curve using trapezoidal integration
        self.AUC = np.trapz(tpr_array, fpr_array)

        mean_fpr = np.mean(self.fpr_list)
        mean_tpr = np.mean(self.tpr_list)
        self.meanROC = (mean_fpr, mean_tpr)
        print(f"Mean FPR: {mean_fpr:.4f}, Mean TPR: {mean_tpr:.4f}, AUC: {self.AUC:.4f}")

    def display_metrics(self):
        print(f"Mean IoU: {self.mIoU:.4f}")
        print(f"Mean F-measure: {self.mFmeasure:.4f}")
        self.plotPR()
        self.plotROC()

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
