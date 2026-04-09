import numpy as np
import matplotlib.pyplot as plt

# 读取
train_loss = np.load("train_loss_history.npy")
train_dice = np.load("train_dice_history.npy")
val_dice = np.load("val_dice_history.npy")

# 画图1 Loss
plt.figure(figsize=(10,4))
plt.plot(train_loss, linewidth=2)
plt.title("Train Loss")
plt.grid(alpha=0.3)
plt.savefig("loss.png", dpi=300)
plt.close()

# 画图2 Dice
plt.figure(figsize=(10,4))
plt.plot(train_dice, label="Train", linewidth=2)
plt.plot(val_dice, label="Val", linewidth=2)
plt.legend()
plt.title("Dice Curve")
plt.grid(alpha=0.3)
plt.savefig("dice.png", dpi=300)
plt.close()