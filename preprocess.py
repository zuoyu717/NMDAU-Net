import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import scipy.ndimage as ndimage
import glob

# 加载配置（完全适配你的目录）
from config import *


# ====================== 工具函数（核心：模糊匹配文件名）======================
def find_file_by_suffix(folder_path, suffix):
    """
    模糊匹配文件：在文件夹中找到 包含指定后缀 的文件
    例如 suffix="_flair.nii" → 自动匹配 "xxx_flair.nii" / "BraTS19_xxx_flair.nii" 等
    """
    # 用glob模糊匹配：* 代表任意前缀
    files = glob.glob(os.path.join(folder_path, f"*{suffix}*"))
    if len(files) == 0:
        raise FileNotFoundError(f"在 {folder_path} 中未找到包含 {suffix} 的文件！")
    if len(files) > 1:
        print(f"⚠️ 警告：{folder_path} 中找到多个 {suffix} 文件，将使用第一个：{files[0]}")
    return files[0]


def load_case(case_folder):
    """
    加载单个病例：自动模糊匹配4个模态+标签，完全适配你的随机前缀文件名
    """
    data = []
    # 加载4个模态（模糊匹配后缀）
    for m in MODALS:
        # 匹配：任意前缀 + _flair.nii / _t1.nii 等
        file_path = find_file_by_suffix(case_folder, f"_{m}.nii")
        img = nib.load(file_path).get_fdata()
        data.append(img)

    # 加载标签seg（匹配任意前缀 + _seg.nii）
    seg_path = find_file_by_suffix(case_folder, "_seg.nii")
    seg = nib.load(seg_path).get_fdata()

    return np.array(data), seg  # data shape: (4, 240, 240, 155)


def crop_background(data, seg):
    """裁剪全黑背景，聚焦肿瘤区域，减少计算量"""
    mask = seg > 0
    if mask.sum() == 0:
        # 无肿瘤时，默认裁剪边缘20个体素
        return data[:, 20:-20, 20:-20, 20:-20], seg[20:-20, 20:-20, 20:-20]
    # 找到肿瘤区域的边界，加16个体素的padding
    coords = np.where(mask)
    z_min = max(0, int(np.min(coords[2])) - 16)
    z_max = min(data.shape[-1], int(np.max(coords[2])) + 16)
    return data[..., z_min:z_max], seg[..., z_min:z_max]


def normalize_modality(data):
    """每个模态独立Z-score归一化，消除扫描设备差异"""
    for i in range(data.shape[0]):
        img = data[i]
        # 只对非零区域（脑组织）做归一化
        brain_mask = img > 0
        if brain_mask.sum() == 0:
            data[i] = 0
            continue
        mean = img[brain_mask].mean()
        std = img[brain_mask].std() + 1e-6  # 加1e-6防止除0
        data[i][brain_mask] = (img[brain_mask] - mean) / std
        data[i][~brain_mask] = 0  # 背景置0
    return data


def resize_to_fixed_size(data, seg, target_size):
    """缩放到统一96³尺寸，适配极速版"""
    # 计算缩放比例：原始240×240×155 → 目标96×96×96
    scale = (
        1,  # 模态维度不变
        target_size[0] / 240,
        target_size[1] / 240,
        target_size[2] / data.shape[3]
    )
    # 图像用3次插值（保留细节），标签用0次插值（保持类别不变）
    data_resized = ndimage.zoom(data, scale, order=3)
    seg_resized = ndimage.zoom(seg, scale[1:], order=0)
    return data_resized, seg_resized


def process_single_case(case_folder):
    """完整处理单个病例：加载→裁剪→归一化→缩放"""
    data, seg = load_case(case_folder)
    data, seg = crop_background(data, seg)
    data = normalize_modality(data)
    data, seg = resize_to_fixed_size(data, seg, IMG_SIZE)
    return data, seg


# ====================== 主函数（适配50病例+你的目录）======================
if __name__ == "__main__":
    # 1. 读取rawdata下所有病例（01-50）
    all_case_folders = [
        os.path.join(RAW_DATA_ROOT, str(i).zfill(2))  # 自动匹配01、02...50
        for i in range(1, 51)
        if os.path.isdir(os.path.join(RAW_DATA_ROOT, str(i).zfill(2)))
    ]

    # 校验病例数（确保是50个）
    print(f"✅ 检测到有效病例数：{len(all_case_folders)} 个")
    if len(all_case_folders) < 50:
        print("⚠️ 警告：rawdata下病例不足50个，请检查01-50文件夹是否完整！")

    # 2. 按7:2:1划分训练/验证/测试集（50例→35/10/5）
    train_cases, test_cases = train_test_split(
        all_case_folders, test_size=TEST_SIZE, random_state=42
    )
    train_cases, val_cases = train_test_split(
        train_cases, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=42
    )
    print(f"📊 数据划分：训练集{len(train_cases)}例 / 验证集{len(val_cases)}例 / 测试集{len(test_cases)}例")

    # 3. 创建输出目录
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DATA_ROOT, split), exist_ok=True)

    # 4. 批量预处理（极速版96³）
    for split_name, cases in zip(["train", "val", "test"], [train_cases, val_cases, test_cases]):
        print(f"\n🚀 正在处理{split_name}集...")
        for case_path in tqdm(cases, desc=f"{split_name}进度"):
            case_name = os.path.basename(case_path)
            # 处理单个病例
            data, seg = process_single_case(case_path)
            # 保存为npy（加速训练）
            np.save(
                os.path.join(DATA_ROOT, split_name, f"{case_name}_data.npy"),
                data.astype(np.float32)  # 用float32进一步省显存
            )
            np.save(
                os.path.join(DATA_ROOT, split_name, f"{case_name}_seg.npy"),
                seg.astype(np.uint8)
            )

    print("\n✅ 极速版数据预处理完成！")
    print(f"📦 输出路径：{DATA_ROOT}")
    print(f"📐 数据尺寸：{IMG_SIZE}（4模态）")