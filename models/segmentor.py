import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeInstanceNorm3d(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.InstanceNorm3d(num_features, affine=affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.size(2) * x.size(3) * x.size(4) <= 1:
            return x
        return self.norm(x)

def _valid_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return max(1, g)

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "in"):
        super().__init__()

        if norm_type == "in":
            norm1 = SafeInstanceNorm3d(out_ch, affine=True)
            norm2 = SafeInstanceNorm3d(out_ch, affine=True)
        elif norm_type == "gn":
            g = _valid_gn_groups(out_ch, max_groups=32)
            norm1 = nn.GroupNorm(num_groups=g, num_channels=out_ch, affine=True)
            norm2 = nn.GroupNorm(num_groups=g, num_channels=out_ch, affine=True)
        elif norm_type == "none":
            norm1 = nn.Identity()
            norm2 = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = norm1
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm2
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x

class UNet3D(nn.Module):
    def __init__(self, in_dim: int = 1, c_dim: int = 0, base_ch: int = 32, out_dim: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.c_dim  = c_dim

        ch = [
            base_ch,
            base_ch * 2,
            base_ch * 4,
            base_ch * 8,
            base_ch * 8,
            base_ch * 8,
            base_ch * 8,
            base_ch * 8,
            base_ch * 8
        ]

        self.enc = nn.ModuleList()
        enc_in = in_dim + c_dim
        for i in range(0, 8):
            self.enc.append(ConvBlock3D(enc_in, ch[i], norm_type="in"))
            enc_in = ch[i]

        self.down = nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True)

        self.bottleneck = ConvBlock3D(ch[7], ch[8], norm_type="gn")

        self.dec = nn.ModuleList()
        dec_in = ch[8]
        for i in reversed(range(0, 8)):
            self.dec.append(ConvBlock3D(dec_in + ch[i], ch[i], norm_type="in"))
            dec_in = ch[i]

        self.head = nn.Conv3d(ch[0], out_dim, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            if isinstance(m, (nn.InstanceNorm3d, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1.)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.)

    @staticmethod
    def _expand_condition(c: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:

        if c.dim() == 2:
            c = c[:, :, None, None, None]
        elif c.dim() != 5:
            raise ValueError(f"Unsupported shape for c: {tuple(c.shape)}")
        return c.expand(-1, -1, D, H, W)

    @staticmethod
    def _maybe_fix_layout(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor, got {x.dim()}D with shape {tuple(x.shape)}")

        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        spatial = [D, H, W]
        min_idx = spatial.index(min(spatial))

        if min_idx == 2:

            x = x.permute(0, 1, 4, 2, 3).contiguous()
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None, return_logits: bool = False) -> torch.Tensor:
        x = self._maybe_fix_layout(x)
        B, C, D, H, W = x.shape

        if self.c_dim > 0:
            if c is None:
                raise ValueError("c must be provided when c_dim > 0")
            c_exp = self._expand_condition(c, D, H, W)
            x = torch.cat([x, c_exp], dim=1)

        feats = []
        z = x
        for i in range(0, 8):
            z = self.enc[i](z)
            feats.append(z)
            if i != 7:
                z = self.down(z)

        z = self.bottleneck(z)

        for di, skip_i in enumerate(range(7, -1, -1)):
            skip = feats[skip_i]
            z = F.interpolate(z, size=skip.shape[2:], mode="trilinear", align_corners=False)
            z = torch.cat([skip, z], dim=1)
            z = self.dec[di](z)

        logits = self.head(z)
        if return_logits:
            return logits
        return torch.sigmoid(logits)
