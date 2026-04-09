import torch
import numpy as np
from medpy import metric

def compute_dice_wt_tc_et(pred, target):
    smooth = 1e-5
    wt_pred = (pred >= 1).float()
    wt_gt = (target >= 1).float()
    wt = (2 * (wt_pred * wt_gt).sum() + smooth) / (wt_pred.sum() + wt_gt.sum() + smooth)

    tc_pred = ((pred == 1) | (pred == 3)).float()
    tc_gt = ((target == 1) | (target == 3)).float()
    tc = (2 * (tc_pred * tc_gt).sum() + smooth) / (tc_pred.sum() + tc_gt.sum() + smooth)

    et_pred = (pred == 3).float()
    et_gt = (target == 3).float()
    et = (2 * (et_pred * et_gt).sum() + smooth) / (et_pred.sum() + et_gt.sum() + smooth)

    return wt.item(), tc.item(), et.item()

def mean_dice(pred, target):
    wt, tc, et = compute_dice_wt_tc_et(pred, target)
    return (wt + tc + et) / 3.0

def accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = torch.numel(pred)
    return correct / total

def auc_roc(pred_prob, target):
    pred_prob = pred_prob[:, 1:].sum(1).view(-1).cpu().numpy()
    target = (target > 0).view(-1).cpu().numpy()
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(target, pred_prob)
    except:
        return 0.0

def hd95(y_pred, y_true):
    try:
        return metric.binary.hd95(y_pred, y_true)
    except Exception as e:
        print(f"[WARN] hd95 计算失败，返回 NaN: {e}")
        return np.nan

def asd(y_pred, y_true):
    try:
        return metric.binary.asd(y_pred, y_true)
    except Exception as e:
        print(f"[WARN] asd 计算失败，返回 NaN: {e}")
        return np.nan