import torch
import numpy as np
from tqdm import tqdm
from utils.dataset import BraTSDataset
from utils.metrics import *
from models.nmdau_net import NMDauNet

# ================== GPU 设置 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ================== 加载测试集 ==================
test_loader = torch.utils.data.DataLoader(
    BraTSDataset("data/test"),
    batch_size=1,
    shuffle=False
)

# ================== 加载最优模型 ==================
model = NMDauNet().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ================== 初始化指标 ==================
total_wt = 0.0
total_tc = 0.0
total_et = 0.0
total_mean = 0.0
total_acc = 0.0
total_auc = 0.0

total_hd_wt = 0.0
total_hd_tc = 0.0
total_hd_et = 0.0
total_asd_wt = 0.0
total_asd_tc = 0.0
total_asd_et = 0.0

num_samples = len(test_loader)

# 空测试集保护：避免后续汇总时除零
if num_samples == 0:
    print("⚠️ 测试集为空：未在 data/test 下找到可评估样本。")
    raise SystemExit(0)

with torch.no_grad():
    for data, seg in tqdm(test_loader, desc="测试中"):
        data = data.to(device)
        seg = seg.to(device)

        output = model(data)
        pred = output.argmax(dim=1)
        prob = torch.softmax(output, dim=1)

        seg_np = seg.cpu().numpy()[0]
        pred_np = pred.cpu().numpy()[0]

        wt, tc, et = compute_dice_wt_tc_et(pred, seg)
        md = mean_dice(pred, seg)
        acc = accuracy(pred, seg)
        auc = auc_roc(prob, seg)

        wt_mask = pred_np >= 1
        wt_gt_mask = seg_np >= 1

        tc_mask = (pred_np == 1) | (pred_np == 3)
        tc_gt_mask = (seg_np == 1) | (seg_np == 3)

        et_mask = pred_np == 3
        et_gt_mask = seg_np == 3

        hd_wt = hd95(wt_mask, wt_gt_mask)
        hd_tc = hd95(tc_mask, tc_gt_mask)
        hd_et = hd95(et_mask, et_gt_mask)

        asd_wt = asd(wt_mask, wt_gt_mask)
        asd_tc = asd(tc_mask, tc_gt_mask)
        asd_et = asd(et_mask, et_gt_mask)

        total_wt += wt
        total_tc += tc
        total_et += et
        total_mean += md
        total_acc += acc
        total_auc += auc

        total_hd_wt += hd_wt
        total_hd_tc += hd_tc
        total_hd_et += hd_et
        total_asd_wt += asd_wt
        total_asd_tc += asd_tc
        total_asd_et += asd_et

# ================== 输出最终结果 ==================
print("\n" + "=" * 70)
print("           BraTS2019 测试集完整指标        ")
print("=" * 70)
print(f"Mean Dice    : {total_mean / num_samples:.4f}")
print(f"WT Dice      : {total_wt / num_samples:.4f}")
print(f"TC Dice      : {total_tc / num_samples:.4f}")
print(f"ET Dice      : {total_et / num_samples:.4f}")
print(f"Accuracy     : {total_acc / num_samples:.4f}")
print(f"AUC-ROC      : {total_auc / num_samples:.4f}")
print(f"HD95-WT      : {total_hd_wt / num_samples:.2f}")
print(f"HD95-TC      : {total_hd_tc / num_samples:.2f}")
print(f"HD95-ET      : {total_hd_et / num_samples:.2f}")
print(f"ASD-WT       : {total_asd_wt / num_samples:.2f}")
print(f"ASD-TC       : {total_asd_tc / num_samples:.2f}")
print(f"ASD-ET       : {total_asd_et / num_samples:.2f}")
print("=" * 70)