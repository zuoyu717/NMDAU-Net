---

## 1. 项目整体结构

```
MICCAI_BraTS_2019_Data_Training/
├── config.py              # 全局超参数配置
├── preprocess.py          # 原始 NIfTI → .npy 预处理
├── models/
│   └── nmdau_net.py       # 模型定义 (NMDauNet)
├── utils/
│   ├── dataset.py         # BraTSDataset
│   └── metrics.py         # Dice/HD95/ASD 等指标
├── train.py               # 训练主流程 + DiceBCELoss
├── test.py                # 测试 + 区域指标 (WT/TC/ET)
├── rawdata/01..50/        # 原始 50 例数据
└── data/{train,val,test}/ # 预处理后的 .npy
```

---

## 2. 配置概览 (`config.py`)

| 项目 | 值 |
|---|---|
| IMG_SIZE | (96, 96, 96) |
| MODALS | ["flair", "t1", "t1ce", "t2"] (4 通道) |
| 类别数 | 4 (0=背景, 1=坏死, 2=水肿, 3=增强；label 4→3 重映射) |
| TRAIN_BATCH_SIZE | 1 |
| TRAIN_EPOCHS | 50 |
| TRAIN_LEARNING_RATE | 1e-4 |
| 数据划分 | 50 例 → train 35 / val 10 / test 5 |

---

## 3. 数据流水线

### 3.1 预处理 `preprocess.py`
对每个病例依次执行：

| 步骤 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 1. 加载 | `load_case()` L28-44 | 病例文件夹 | `data [4,240,240,155]`, `seg [240,240,155]` |
| 2. 裁剪背景 | `crop_background()` L47-57 | 上一步 | `data [4,H',W',D']`, `seg [H',W',D']` (肿瘤 ROI + 16 voxel padding) |
| 3. 模态内 Z-score | `normalize_modality()` L60-73 | 上一步 | 形状不变，仅对非零脑组织归一化 |
| 4. Resize | `resize_to_fixed_size()` L76-88 | 上一步 | `data [4,96,96,96]` (cubic), `seg [96,96,96]` (nearest) |

保存为 `{case}_data.npy` (float32) 与 `{case}_seg.npy` (uint8)。

### 3.2 Dataset / DataLoader `utils/dataset.py`
`BraTSDataset.__getitem__`:
- 读 `.npy` → 标签重映射 `seg[seg==4]=3`
- 返回:
  - `data: FloatTensor [4, 96, 96, 96]`
  - `seg : LongTensor  [96, 96, 96]`

DataLoader batch (batch_size=1):
- `data: [1, 4, 96, 96, 96]`
- `seg : [1, 96, 96, 96]`

---

## 4. 模型架构：NMDauNet (`models/nmdau_net.py`)

**整体：** 3 层编码 / ASPP 瓶颈 / 2 层解码的轻量 3D U-Net，基础通道 `c=16`，大量使用 Depthwise Separable Conv、双注意力 (DAM)、ASPP 和 BiFPN 跳连融合。

### 4.1 基础构件

#### (a) `DepthwiseSeparableConv3D` (L8-19)
轻量 3D 卷积块：Depthwise 3×3×3 → Pointwise 1×1×1 → BN → ReLU
- 输入 `[B, C_in, H, W, D]` → 输出 `[B, C_out, H, W, D]` (空间尺寸不变)

#### (b) `DAM` 双注意力模块 (L24-40)
- **通道注意力 CA**: AvgPool+MaxPool → `[B,C,1,1,1]` → Conv(C→C/16)→ReLU→Conv(C/16→C) → Sigmoid
- **空间注意力 SA**: 沿通道取 mean & max → `[B,2,H,W,D]` → Conv(2→1, 7³) → Sigmoid
- 输出 = 输入 × CA × SA，形状不变 `[B,C,H,W,D]`

#### (c) `ASPP` (L45-53) - 瓶颈多尺度提取
两条并行 DepthwiseSeparableConv3D → concat → `[B, 2·C_out, H, W, D]` → DepthwiseSeparableConv3D 融合 → `[B, C_out, H, W, D]`

#### (d) `BiFPN` (L58-64) - 跳连融合
高层特征 trilinear 上采样到低层尺度 → 按位相加 → DepthwiseSeparableConv3D
- 输入 `feat1 [B,C,H1,W1,D1]`, `feat2 [B,C,H2,W2,D2]`
- 输出 `[B, C, H1, W1, D1]`

### 4.2 NMDauNet 前向流程 (L69-113)

```
Input x:                       [B, 4, 96, 96, 96]

── Encoder ───────────────────────────────────
e1  = DSConv(4→16)(x)          [B, 16, 96, 96, 96]
e1  = DAM(16)(e1)              [B, 16, 96, 96, 96]
p1  = MaxPool3d(2)(e1)         [B, 16, 48, 48, 48]

e2  = DSConv(16→32)(p1)        [B, 32, 48, 48, 48]
e2  = DAM(32)(e2)              [B, 32, 48, 48, 48]
p2  = MaxPool3d(2)(e2)         [B, 32, 24, 24, 24]

e3  = DSConv(32→64)(p2)        [B, 64, 24, 24, 24]
e3  = DAM(64)(e3)              [B, 64, 24, 24, 24]
p3  = MaxPool3d(2)(e3)         [B, 64, 12, 12, 12]

── Bottleneck ────────────────────────────────
b   = ASPP(64→64)(p3)          [B, 64, 12, 12, 12]

── Decoder (BiFPN 跳连) ───────────────────────
f3  = BiFPN(64)(e3, b)         [B, 64, 24, 24, 24]
d3  = DSConv(64→32)(f3)        [B, 32, 24, 24, 24]

f2  = BiFPN(32)(e2, d3)        [B, 32, 48, 48, 48]
d2  = DSConv(32→16)(f2)        [B, 16, 48, 48, 48]

── Head ──────────────────────────────────────
u   = Trilinear(d2, size=96)   [B, 16, 96, 96, 96]
out = Conv3d(16→4, 1³)(u)      [B,  4, 96, 96, 96]   (logits)
```

**通道演化：** 4 → 16 → 32 → 64 → (ASPP) → 32 → 16 → 4
**空间演化：** 96 → 48 → 24 → 12 → 24 → 48 → 96

---

## 5. 损失函数 `DiceBCELoss` (`train.py` L26-40)

输入：`pred [B,4,96,96,96]` logits, `target [B,96,96,96]` 类别索引
- CE = `CrossEntropyLoss(pred, target)` (4 类)
- Dice: 对 softmax(pred) 的 **类 1/2/3** (忽略背景) 计算
  - `dice_c = 1 - (2·∩ + ε) / (Σp + Σt + ε)`
  - 对三类平均 → `Dice/3`
- **Total = CE + Dice/3**  (scalar)

---

## 6. 训练与测试流程

### 6.1 `train.py`
- Optimizer: `Adam(lr=1e-4, weight_decay=1e-5)`
- 每 epoch：train forward → loss backward → step；验证集计算 `mean_dice`，保存最优 `best_model.pth`
- 历史记录保存为 `train_loss_history.npy`, `train_dice_history.npy`, `val_dice_history.npy`

### 6.2 `test.py` 指标计算
对每个测试样本：
- `out = model(data)` → `[1,4,96,96,96]`
- `pred = out.argmax(1)` → `[1,96,96,96]`；`prob = softmax(out,1)`
- **区域划分**: WT(`pred≥1`), TC(`pred∈{1,3}`), ET(`pred==3`)
- 指标 (`utils/metrics.py`):
  - `compute_dice_wt_tc_et` / `mean_dice`
  - `accuracy`, `auc_roc`
  - `hd95`, `asd` (依赖 medpy)

---

## 7. 关键张量形状速查表

| 阶段 | 张量 | 形状 |
|---|---|---|
| 原始 NIfTI | 单模态 | (240, 240, 155) |
| 预处理后 .npy | data / seg | (4,96,96,96) / (96,96,96) |
| DataLoader 输出 | data / seg | [1,4,96,96,96] / [1,96,96,96] |
| Encoder e1/e2/e3 | 特征 | [B,16,96³] / [B,32,48³] / [B,64,24³] |
| ASPP 瓶颈 | 特征 | [B,64,12,12,12] |
| Decoder d3/d2 | 特征 | [B,32,24³] / [B,16,48³] |
| 模型输出 logits | out | [B,4,96,96,96] |
| 预测类别 | argmax | [B,96,96,96] |

---

## 8. 架构亮点小结
1. **轻量化**：Depthwise Separable Conv + 基础通道仅 16，可在 8GB 显存下以 96³ patch 训练。
2. **注意力**：编码器每层后接 DAM (通道 + 空间注意力)。
3. **多尺度**：瓶颈处 ASPP 并行分支融合多尺度上下文。
4. **双向特征融合**：解码跳连使用 BiFPN (加权相加 + DSConv) 而非传统 concat。
5. **损失**：CE + 前景 Dice 组合，缓解类别不平衡。

---
