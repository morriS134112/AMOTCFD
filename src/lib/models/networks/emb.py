import torch
from torch import nn
from lib.models.decode import mot_decode

class EmbeddingLayer(nn.Module):
    def __init__(self, channel=128, S=4, Groups=8):
        super(EmbeddingLayer, self).__init__()
        self.S = S  # 分組數
        self.G = Groups  # 通道分組數

        # 建立多個卷積層，卷積核大小隨著索引增大
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1) for i in range(S)]
        )
        self.softmax = nn.Softmax(-1)  # 定義softmax層，用於計算權重
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 自適應平均池化，輸出尺寸為1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 自適應平均池化，輸出高度不變，寬度為1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 自適應平均池化，輸出高度為1，寬度不變
        self.gn = nn.GroupNorm(channel // self.G, channel // self.G)  # 群組歸一化
        self.conv1x1 = nn.Conv2d(channel // self.G, channel // self.G, kernel_size=1, stride=1, padding=0)  # 1x1卷積
        self.conv3x3 = nn.Conv2d(channel // self.G, channel // self.G, kernel_size=3, stride=1, padding=1)  # 3x3卷積

    def channel_shuffle(self, x, groups):
        # 通道混洗，用於重新排列通道以增強特徵融合
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w).permute(0, 2, 1, 3, 4).contiguous().view(b, -1, h, w)
        return x

    def forward(self, x, hm=None, wh=None, reg=None):
        b, c, h, w = x.size()
        # 進行多物體檢測，得到檢測結果與索引
        detections, inds = mot_decode(hm, wh, reg=reg)
        detections = detections[:, :, :4].int()  # 只保留前四個檢測結果（x1, y1, x2, y2）

        group_x = x.reshape(b * self.G, -1, h, w)  # 將輸入重新排列為分組形式
        x_h = self.pool_h(group_x)  # 對每個分組進行高度方向的平均池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 對每個分組進行寬度方向的平均池化，並轉置
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 將高度和寬度池化結果拼接後進行1x1卷積
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 將卷積結果分割為高度和寬度部分
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 將分組輸入與sigmoid激活的高度和寬度結果相乘，並進行群組歸一化
        x2 = self.conv3x3(group_x)  # 對分組輸入進行3x3卷積
        x11 = self.softmax(self.agp(x1).reshape(b * self.G, -1, 1).permute(0, 2, 1))  # 對x1進行自適應平均池化，並計算softmax權重
        x12 = x2.reshape(b * self.G, c // self.G, -1)  # 將x2重新排列為分組形式
        x21 = self.softmax(self.agp(x2).reshape(b * self.G, -1, 1).permute(0, 2, 1))  # 對x2進行自適應平均池化，並計算softmax權重
        x22 = x1.reshape(b * self.G, c // self.G, -1)  # 將x1重新排列為分組形式
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.G, 1, h, w)  # 計算權重矩陣，並重新排列為分組形式
        SPC_out = (group_x * weights.sigmoid()).view(b, self.S, c // self.S, h, w)

        # 動態感受野選擇與多尺度融合
        for idx, conv in enumerate(self.convs):
            # 根據目標密集程度進行自適應感受野調整
            region_size = 5  # 預設 5x5，但這部分可以動態選擇
            half_size = region_size // 2

            center_x = detections[:, idx, 0]
            center_y = detections[:, idx, 1]

            center_region = torch.zeros((b, c // self.S, region_size, region_size), device=x.device)
            for i in range(b):
                if half_size <= center_y[i] < h - half_size and half_size <= center_x[i] < w - half_size:
                    center_region[i] = SPC_out[i, idx, :, center_y[i] - half_size:center_y[i] + half_size + 1,
                                               center_x[i] - half_size:center_x[i] + half_size + 1].clone()

            # 使用對應的卷積層進行卷積操作
            center_region = conv(center_region)

            # 將卷積後的結果放回原特徵圖中的對應位置
            for i in range(b):
                if half_size <= center_y[i] < h - half_size and half_size <= center_x[i] < w - half_size:
                    SPC_out[i, idx, :, center_y[i] - half_size:center_y[i] + half_size + 1,
                            center_x[i] - half_size:center_x[i] + half_size + 1] = center_region[i].clone()

        # 應用sigmoid激活函數，生成注意力圖
        sigmoid_out = torch.sigmoid(SPC_out)

        # 確保 SPC_out 和 sigmoid_out 的形狀與 x 匹配
        SPC_out = SPC_out.view(b, c, h, w)
        sigmoid_out = sigmoid_out.view(b, c, h, w)

        # 將原始特徵圖與注意力圖相乘
        out = x + (SPC_out * sigmoid_out)

        # 進行通道混洗以增加特徵的表現力
        out = self.channel_shuffle(out, self.G)

        return out