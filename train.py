import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import numpy as np  # <-- 只加这一句

from utils.dataset import BraTSDataset
from utils.metrics import mean_dice
from models.nmdau_net import NMDauNet
import config as cfg

torch.multiprocessing.set_sharing_strategy('file_system')

# 通过 config 读取训练参数；若未配置则保持当前脚本原有常量
TRAIN_BATCH_SIZE = getattr(cfg, "TRAIN_BATCH_SIZE", 1)
TRAIN_EPOCHS = getattr(cfg, "TRAIN_EPOCHS", 50)
TRAIN_LEARNING_RATE = getattr(cfg, "TRAIN_LEARNING_RATE", 1e-4)
TRAIN_NUM_WORKERS = getattr(cfg, "TRAIN_NUM_WORKERS", 0)

# ==========================================================================
# 论文损失：Dice + BCE
# ==========================================================================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, pred, target):
        ce = self.ce(pred, target)
        pred = torch.softmax(pred, dim=1)
        smooth = 1e-5
        dice = 0
        for c in range(1,4):
            p = pred[:,c]
            t = (target==c).float()
            inter = (p*t).sum()
            dice += 1 - (2*inter+smooth)/(p.sum()+t.sum()+smooth)
        return ce + dice/3

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(BraTSDataset("data/train"), batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=TRAIN_NUM_WORKERS)
    val_loader = DataLoader(BraTSDataset("data/val"), batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=TRAIN_NUM_WORKERS)

    model = NMDauNet().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LEARNING_RATE, weight_decay=1e-5)

    best_val = 0

    # ======================== 用于绘图的列表 ========================
    train_loss_history = []
    train_dice_history = []
    val_dice_history = []

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        loss_sum = dice_sum = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for data, seg in pbar:
            data, seg = data.to(device), seg.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, seg)
            loss.backward()
            optimizer.step()
            dice = mean_dice(out.argmax(1), seg)
            loss_sum += loss.item()
            dice_sum += dice
            pbar.set_postfix(loss=loss.item(), dice=dice)

        # 记录平均指标
        avg_loss = loss_sum / len(train_loader)
        avg_dice = dice_sum / len(train_loader)

        train_loss_history.append(avg_loss)
        train_dice_history.append(avg_dice)

        # 验证
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for data, seg in val_loader:
                data, seg = data.to(device), seg.to(device)
                val_dice += mean_dice(model(data).argmax(1), seg)
        val_dice /= len(val_loader)
        print(f"Val Dice: {val_dice:.4f}")

        val_dice_history.append(val_dice)

        if val_dice > best_val:
            best_val = val_dice
            torch.save(model.state_dict(), "best_model.pth")

    # ======================== 只加这 3 行：保存指标给 plot.py ========================
    np.save("train_loss_history.npy", np.array(train_loss_history))
    np.save("train_dice_history.npy", np.array(train_dice_history))
    np.save("val_dice_history.npy", np.array(val_dice_history))