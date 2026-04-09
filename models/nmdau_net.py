import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# 1. 深度可分离卷积 3D
# -------------------------------------------------------------------------
class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv3d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))

# -------------------------------------------------------------------------
# 2. DAM 注意力模块
# -------------------------------------------------------------------------
class DAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels//16, 1),
            nn.ReLU(),
            nn.Conv3d(channels//16, channels, 1)
        )
        self.spatial = nn.Conv3d(2, 1, 7, padding=3)

    def forward(self, x):
        ca = torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1,True), x.max(1,True)[0]], 1)))
        return x * sa

# -------------------------------------------------------------------------
# 3. ASPP 多尺度
# -------------------------------------------------------------------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv3D(in_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv3D(in_ch, out_ch)
        self.out = DepthwiseSeparableConv3D(out_ch*2, out_ch)

    def forward(self, x):
        return self.out(torch.cat([self.conv1(x), self.conv2(x)], 1))

# -------------------------------------------------------------------------
# 4. BiFPN 双向融合
# -------------------------------------------------------------------------
class BiFPN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = DepthwiseSeparableConv3D(c, c)
    def forward(self, feat1, feat2):
        feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(feat1 + feat2)

# -------------------------------------------------------------------------
# 5. ✅ 最终修复版 NMDAU-Net（通道完全对齐）
# -------------------------------------------------------------------------
class NMDauNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        c = 16

        # Encoder
        self.e1 = DepthwiseSeparableConv3D(in_channels, c)
        self.dam1 = DAM(c)

        self.e2 = DepthwiseSeparableConv3D(c, c*2)
        self.dam2 = DAM(c*2)

        self.e3 = DepthwiseSeparableConv3D(c*2, c*4)
        self.dam3 = DAM(c*4)

        self.pool = nn.MaxPool3d(2)
        self.aspp = ASPP(c*4, c*4)

        # Decoder
        self.bifpn3 = BiFPN(c*4)
        self.up3 = DepthwiseSeparableConv3D(c*4, c*2)

        self.bifpn2 = BiFPN(c*2)
        self.up2 = DepthwiseSeparableConv3D(c*2, c)  # ✅ 这里确保输出 16 通道

        # 输出层
        self.out = nn.Conv3d(c, num_classes, 1)

    def forward(self, x):
        # 编码
        e1 = self.dam1(self.e1(x))
        e2 = self.dam2(self.e2(self.pool(e1)))
        e3 = self.dam3(self.e3(self.pool(e2)))

        # 瓶颈
        b = self.aspp(self.pool(e3))

        # 解码
        d3 = self.bifpn3(e3, b)
        d2 = self.bifpn2(e2, self.up3(d3))
        d2 = self.up2(d2)  # ✅ 强制变成 16 通道

        # 最终输出
        out = F.interpolate(d2, size=x.shape[2:], mode="trilinear", align_corners=False)
        return self.out(out)