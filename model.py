import torch
import torch.nn as nn
from loss import LossFunction, TextureDifference
from utils import blur, pair_downsampler


# basic block

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# denoiser

class Denoise_1(nn.Module):
    def __init__(self, chan_embed=48):
        super().__init__()

        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.res1 = ResidualBlock(chan_embed)
        self.res2 = ResidualBlock(chan_embed)

        self.attn = ChannelAttention(chan_embed)
        self.conv_out = nn.Conv2d(chan_embed, 3, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        x = self.conv_out(x)
        return x


class Denoise_2(nn.Module):
    def __init__(self, chan_embed=48):
        super().__init__()

        self.conv1 = nn.Conv2d(6, chan_embed, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.res1 = ResidualBlock(chan_embed)
        self.res2 = ResidualBlock(chan_embed)

        self.attn = ChannelAttention(chan_embed)
        self.conv_out = nn.Conv2d(chan_embed, 6, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        x = self.conv_out(x)
        return x


# enhancer

class Enhancer(nn.Module):
    def __init__(self, layers, channels):
        super().__init__()

        self.in_conv = nn.Conv2d(3, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(layers)]
        )

        self.attn = ChannelAttention(channels)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu(self.in_conv(x))
        x = self.blocks(x)
        x = self.attn(x)
        x = self.out_conv(x)
        return torch.clamp(x, 1e-4, 1)


# main network

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.enhance = Enhancer(layers=3, channels=64)
        self.denoise_1 = Denoise_1(chan_embed=48)
        self.denoise_2 = Denoise_2(chan_embed=48)

        self._criterion = LossFunction()
        self.TextureDifference = TextureDifference()

    def forward(self, input):
        eps = 1e-4
        input = input + eps

        L11, L12 = pair_downsampler(input)
        L_pred1 = L11 - self.denoise_1(L11)
        L_pred2 = L12 - self.denoise_1(L12)

        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)

        s2 = self.enhance(L2.detach())
        s21, s22 = pair_downsampler(s2)

        H2 = torch.clamp(input / s2, eps, 1)

        H11 = torch.clamp(L11 / s21, eps, 1)
        H12 = torch.clamp(L12 / s22, eps, 1)

        H3_pred = torch.cat([H11, s21], 1).detach() - self.denoise_2(torch.cat([H11, s21], 1))
        H3_pred = torch.clamp(H3_pred, eps, 1)

        H13, s13 = H3_pred[:, :3], H3_pred[:, 3:]

        H4_pred = torch.cat([H12, s22], 1).detach() - self.denoise_2(torch.cat([H12, s22], 1))
        H4_pred = torch.clamp(H4_pred, eps, 1)

        H14, s14 = H4_pred[:, :3], H4_pred[:, 3:]

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)

        H3, s3 = H5_pred[:, :3], H5_pred[:, 3:]

        L_diff = self.TextureDifference(L_pred1, L_pred2)

        H3_d1, H3_d2 = pair_downsampler(H3)
        H3_diff = self.TextureDifference(H3_d1, H3_d2)

        H1 = torch.clamp(L2 / s2, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)

        return (L_pred1, L_pred2, L2, s2, s21, s22, H2,
                H11, H12, H13, s13, H14, s14,
                H3, s3, H3_pred, H4_pred,
                L_diff, H3_diff, H2_blur, H3_blur)

    def _loss(self, input):
        outputs = self(input)
        return self._criterion(input, *outputs)


# finetuning

class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super().__init__()

        self.enhance = Enhancer(layers=3, channels=64)
        self.denoise_1 = Denoise_1(chan_embed=48)
        self.denoise_2 = Denoise_2(chan_embed=48)

        pretrained = torch.load(weights, map_location='cuda:0')
        self.load_state_dict({k: v for k, v in pretrained.items() if k in self.state_dict()}, strict=False)

    def forward(self, input):
        eps = 1e-4
        input = input + eps

        L2 = torch.clamp(input - self.denoise_1(input), eps, 1)
        s2 = self.enhance(L2)

        H2 = torch.clamp(input / s2, eps, 1)

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)

        H3 = H5_pred[:, :3]

        return H2, H3