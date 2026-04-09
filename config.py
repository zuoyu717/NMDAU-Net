# ========================
# RTX 5060 8GB 极速版配置（适配你的目录+50病例）
# ========================

# ✅ 你的原始数据路径（data同级的rawdata）
RAW_DATA_ROOT = "rawdata"
# ✅ 预处理后数据输出路径
DATA_ROOT = "data"

# 极速核心：图像尺寸 96³（提速2倍+，8GB显存稳跑）
IMG_SIZE = (96, 96, 96)

# 训练参数（绝对不爆显存）
BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 1
PIN_MEMORY = False

# train.py 使用的训练参数（保持与当前 train.py 常量一致）
TRAIN_BATCH_SIZE = 1
TRAIN_EPOCHS = 50
TRAIN_LEARNING_RATE = 1e-4
TRAIN_NUM_WORKERS = 0

# 模态顺序（固定不变，和你的nii.gz文件名对应）
MODALS = ["flair", "t1", "t1ce", "t2"]

# ✅ 你要求：只使用50个病例（rawdata下01-50）
USE_SUBSET = True
SUBSET_SIZE = 50

# 数据划分比例（50例按7:2:1分：35 train / 10 val / 5 test）
TEST_SIZE = 0.2
VAL_SIZE = 0.1