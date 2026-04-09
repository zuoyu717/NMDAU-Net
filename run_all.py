import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(step_name, cmd):
    print("\n" + "=" * 80)
    print(f"[RUN] {step_name}: {' '.join(cmd)}")
    print("=" * 80)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAIL] {step_name} 失败，退出码: {result.returncode}")
        sys.exit(result.returncode)
    print(f"[OK] {step_name} 完成")


def assert_exists(path, desc):
    if not Path(path).exists():
        print(f"[ERROR] 缺少{desc}: {path}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键运行：预处理->训练->可视化->测试")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过 preprocess.py")
    parser.add_argument("--skip-visualize", action="store_true", help="跳过 visualize.py")
    args = parser.parse_args()

    py = sys.executable  # 当前Python解释器

    # 基础文件检查
    for f in ["train.py", "test.py", "plot.py", "visualize.py", "config.py"]:
        assert_exists(f, "脚本文件")

    # 1) 预处理（可选）
    if not args.skip_preprocess:
        run_step("Preprocess", [py, "preprocess.py"])
    else:
        print("[SKIP] 已跳过 preprocess.py")

    # 2) 训练
    run_step("Train", [py, "train.py"])

    # 3) 训练曲线可视化
    # train.py 产出 train_loss_history.npy / train_dice_history.npy / val_dice_history.npy
    for f in ["train_loss_history.npy", "train_dice_history.npy", "val_dice_history.npy"]:
        assert_exists(f, "训练历史文件")
    run_step("Plot Curves", [py, "plot.py"])

    # 4) 单病例可视化（可选）
    if not args.skip_visualize:
        assert_exists("best_model.pth", "最佳模型")
        run_step("Visualize", [py, "visualize.py"])
    else:
        print("[SKIP] 已跳过 visualize.py")

    # 5) 测试集评估
    assert_exists("best_model.pth", "最佳模型")
    run_step("Test", [py, "test.py"])

    print("\n" + "=" * 80)
    print("全部流程完成。输出文件通常包括：")
    print("- best_model.pth")
    print("- loss.png")
    print("- dice.png")
    print("- result.png")
    print("- 控制台中的 test 指标（Mean Dice / WT / TC / ET / HD95 / ASD 等）")
    print("=" * 80)