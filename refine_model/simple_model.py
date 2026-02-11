""" Full assembly of the parts to form the complete network """


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):


    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=True, padding_mode='reflect'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, bias=True, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode='reflect')

    def forward(self, x):
        return self.conv(x)

class SimpleNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_indices, bilinear=False):
        super(SimpleNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.embedding = nn.Embedding(num_indices, 10)

        self.inc = (DoubleConv(n_channels, 64))
        self.mid_conv = (DoubleConv(64, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, indices):
        B = x.shape[0]
        H = x.shape[2]
        W = x.shape[3]
        idx_emb = self.embedding(indices)
        idx_emb = idx_emb.view(B, -1, 1, 1)
        idx_emb = idx_emb.expand(-1, -1, H, W)

        x = torch.cat([x, idx_emb], dim=1)

        x1 = self.inc(x)
        x2 = self.mid_conv(x1)
        logits = self.outc(x2)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.mid_conv = torch.utils.checkpoint(self.mid_conv)
        self.outc = torch.utils.checkpoint(self.outc)