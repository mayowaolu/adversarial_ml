import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- basic pre‑activation block ------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, drop_rate=0.0):
        super().__init__()
        self.equal_io = (in_ch == out_ch)

        self.bn1   = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        self.drop_rate = drop_rate
        self.shortcut  = None if self.equal_io else nn.Conv2d(
            in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x)) if self.equal_io else x
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return (x if self.equal_io else self.shortcut(x)) + out


# -------- stack of BasicBlocks -----------------------------------------------
class NetworkBlock(nn.Module):
    def __init__(self, n_layers, in_ch, out_ch, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(
                BasicBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    stride if i == 0 else 1,
                    drop_rate,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# -------- Wide ResNet ---------------------------------------------------------
class WideResNet(nn.Module):
    """
    Wide ResNet (WRN‑d‑k)        – Zagoruyko & Komodakis, 2016
    depth d = 6n+4, widen factor k.

    Example: WRN‑34‑10  -> depth=34, widen_factor=10
    """
    def __init__(self, depth=34, widen_factor=10, num_classes=10, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n = (depth - 4) // 6

        ch = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1  = nn.Conv2d(3, ch[0], 3, padding=1, bias=False)
        self.block1 = NetworkBlock(n, ch[0], ch[1], stride=1, drop_rate=drop_rate)
        self.block2 = NetworkBlock(n, ch[1], ch[2], stride=2, drop_rate=drop_rate)
        self.block3 = NetworkBlock(n, ch[2], ch[3], stride=2, drop_rate=drop_rate)

        self.bn     = nn.BatchNorm2d(ch[3])
        self.relu   = nn.ReLU(inplace=True)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(ch[3], num_classes)

        self._init_weights()

    # --------------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # --------------------------------------------------------------------------
    def forward(self, x):
        x = self.conv1(x)          # 32×32 → 32×32
        x = self.block1(x)         # 32×32
        x = self.block2(x)         # 16×16
        x = self.block3(x)         # 8×8
        x = self.relu(self.bn(x))
        x = self.pool(x).flatten(1)  # global avg‑pool
        return self.fc(x)



# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
#         super(BasicBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.droprate = dropRate
#         self.equalInOut = (in_planes == out_planes)
#         self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
#                                                                 padding=0, bias=False) or None

#     def forward(self, x):
#         if not self.equalInOut:
#             x = self.relu1(self.bn1(x))
#         else:
#             out = self.relu1(self.bn1(x))
#         out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
#         out = self.conv2(out)
#         return torch.add(x if self.equalInOut else self.convShortcut(x), out)


# class NetworkBlock(nn.Module):
#     def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
#         super(NetworkBlock, self).__init__()
#         self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

#     def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
#         layers = []
#         for i in range(int(nb_layers)):
#             layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         return self.layer(x)


# class WideResNet(nn.Module):
#     def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
#         super(WideResNet, self).__init__()
#         nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
#         assert ((depth - 4) % 6 == 0)
#         n = (depth - 4) / 6
#         block = BasicBlock
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # 1st block
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 1st sub-block
#         self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 2nd block
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         # 3rd block
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
#         self.nChannels = nChannels[3]

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         out = out.view(-1, self.nChannels)
#         return self.fc(out)

# # def WideResNet_28_10():
# #     return WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)

# # def WideResNet_28_10_cifar100():
# #     return WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.0)

# # def WideResNet_34_10_CIFAR10():
# #     return WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

# # def WideResNet_34_10_CIFAR100():
# #     return WideResNet(depth=34, num_classes=100, widen_factor=10, dropRate=0.0)

# # def WideResNet_34_10_CIFAR100_P(num_classes):
# #     return WideResNet(depth=34, num_classes=num_classes, widen_factor=10, dropRate=0.0)

# # def WideResNet_16_8_SVHN():
# #     return WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.0)