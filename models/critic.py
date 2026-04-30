import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as _SN


def maybe_sn(module: nn.Module, use_sn: bool) -> nn.Module:
    return _SN(module) if use_sn else module


class ResDownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = False):
        super().__init__()
        self.conv1 = maybe_sn(nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True), use_sn)
        self.conv2 = maybe_sn(nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), use_sn)

        self.skip_pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.skip_conv = maybe_sn(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True), use_sn)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)

        s = self.skip_pool(x)
        s = self.skip_conv(s)

        return self.act(h + s)


class ResBlock3D(nn.Module):
    def __init__(self, ch: int, use_sn: bool = False):
        super().__init__()
        self.conv1 = maybe_sn(nn.Conv3d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True), use_sn)
        self.conv2 = maybe_sn(nn.Conv3d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True), use_sn)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class PatchGAN_Critic(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 64, num_classes: int = 3, use_sn: bool = False):
        super().__init__()
        self.use_sn = use_sn

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 8

        self.down1 = ResDownBlock3D(in_channels, c1, use_sn=use_sn)
        self.down2 = ResDownBlock3D(c1, c2, use_sn=use_sn)
        self.down3 = ResDownBlock3D(c2, c3, use_sn=use_sn)
        self.down4 = ResDownBlock3D(c3, c4, use_sn=use_sn)
        self.down5 = ResDownBlock3D(c4, c5, use_sn=use_sn)

        self.refine = nn.Sequential(
            maybe_sn(nn.Conv3d(c5, c5, kernel_size=3, stride=1, padding=1, bias=True), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock3D(c5, use_sn=use_sn),
        )

        self.conv_out = maybe_sn(nn.Conv3d(c5, 1, kernel_size=1, stride=1, padding=0, bias=True), use_sn)
        self.fc_cls   = maybe_sn(nn.Linear(c5, num_classes, bias=True), use_sn)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, cond):
        h = torch.cat([x, cond], dim=1)

        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.down5(h)

        h = self.refine(h)

        adv_out = self.conv_out(h)
        feat    = torch.mean(h, dim=[2, 3, 4])
        cls_out = self.fc_cls(feat)
        return adv_out, cls_out

