import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import BraTSDataset
from models.nmdau_net import NMDauNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NMDauNet().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

ds = BraTSDataset("data/test")
data, seg = ds[0]  # 取第1个病例
data = data.unsqueeze(0).to(device)

with torch.no_grad():
    out = model(data)
    pred = out.argmax(1).squeeze().cpu().numpy()

slice_idx = 48  # 切中间一层
img = data[0, 0].squeeze().cpu().numpy()  # flair模态
seg = seg.numpy()

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.title("Image")
plt.imshow(img[..., slice_idx], cmap="gray")

plt.subplot(132)
plt.title("Ground Truth")
plt.imshow(seg[..., slice_idx])

plt.subplot(133)
plt.title("Prediction")
plt.imshow(pred[..., slice_idx])

plt.savefig("result.png")
plt.show()