import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

SN = spectral_norm 

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

        self.proj = None
        if out_channels != in_channels:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        out = self.act(out)
        return out

class SelfAttn3D_old_version(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        c_qk = max(in_channels // 8, 1)
        c_v  = max(in_channels // 2, 1)
        self.query_conv = nn.Conv3d(in_channels, c_qk, kernel_size=1, bias=False)
        self.key_conv   = nn.Conv3d(in_channels, c_qk, kernel_size=1, bias=False)
        self.value_conv = nn.Conv3d(in_channels, c_v,  kernel_size=1, bias=False)
        self.out_conv   = nn.Conv3d(c_v, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W

        q = self.query_conv(x).view(B, -1, N)            # (B, Cq, N)
        k = self.key_conv(x).view(B, -1, N)              # (B, Cq, N)
        attn = torch.bmm(q.permute(0, 2, 1), k)          # (B, N, N)
        attn = F.softmax(attn, dim=-1)

        v = self.value_conv(x).view(B, -1, N)            # (B, Cv, N)
        out = torch.bmm(v, attn).view(B, -1, D, H, W)
        out = self.out_conv(out)

        return self.gamma * out + x


class SelfAttn3D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        c_qk = max(in_channels // 8, 1)
        c_v  = max(in_channels // 2, 1)

        self.query_conv = nn.Conv3d(in_channels, c_qk, kernel_size=1, bias=False)
        self.key_conv   = nn.Conv3d(in_channels, c_qk, kernel_size=1, bias=False)
        self.value_conv = nn.Conv3d(in_channels, c_v,  kernel_size=1, bias=False)
        self.out_conv   = nn.Conv3d(c_v, in_channels, kernel_size=1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = (c_qk ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W
        q = self.query_conv(x).view(B, -1, N).transpose(1, 2).contiguous()
        k = self.key_conv(x).view(B, -1, N).contiguous()

        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        v = self.value_conv(x).view(B, -1, N).contiguous()

        # IMPORTANT: use attn^T here
        out = torch.bmm(v, attn.transpose(1, 2).contiguous())
        out = out.view(B, -1, D, H, W)
        out = self.out_conv(out)

        return self.gamma * out + x
class AttnGate3D(nn.Module):
    def __init__(self, skip_channels, gating_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)

        self.theta = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True)
        self.phi   = nn.Conv3d(gating_channels, inter_channels, kernel_size=1, bias=True)
        self.psi   = nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True)

        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_skip, x_gating):
        theta_x = self.theta(x_skip)
        phi_g   = self.phi(x_gating)

        if theta_x.shape[2:] != phi_g.shape[2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode="trilinear", align_corners=False)

        f = self.act(theta_x + phi_g)
        psi = self.sigmoid(self.psi(f))
        return x_skip * psi



class SelfAttentionBlock(nn.Module):
    """
    Memory-Bounded Hybrid Attention Block
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.res_scale = 0.2
        self.se_scale  = 0.2
        c = channels
        c_int = max(c // reduction, 8)

        self.max_q_tokens   = 20_000
        self.max_kv_tokens  = 5_000
        self.max_attn_elems = 32_000_000

        self.theta = SN(nn.Conv3d(c, c_int, 1, bias=False))
        self.phi   = SN(nn.Conv3d(c, c_int, 1, bias=False))
        self.g     = SN(nn.Conv3d(c, c_int, 1, bias=False))
        self.out   = SN(nn.Conv3d(c_int, c, 1, bias=False))

        self.avg = nn.AdaptiveAvgPool3d(1)
        self.se  = nn.Sequential(
            nn.Conv3d(c, c_int, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_int, c, 1, bias=False),
            nn.Sigmoid(),
        )

        self._scale = 1.0 / math.sqrt(float(c_int))

    @staticmethod
    def _stride_for_limit(D, H, W, max_tokens: int) -> int:
        s = 1
        def n_after(_s):
            return max(1, D // _s) * max(1, H // _s) * max(1, W // _s)
        while n_after(s) > max_tokens and min(D // (s + 1), H // (s + 1), W // (s + 1)) >= 1:
            s += 1
        return s

    def forward(self, x):
        B, C, D, H, W = x.shape
        N_full = D * H * W

        w = self.se(self.avg(x))

        if N_full > self.max_q_tokens * 8:
            return x + self.se_scale * (x * w)

        s_q  = self._stride_for_limit(D, H, W, self.max_q_tokens)
        s_kv = self._stride_for_limit(D, H, W, self.max_kv_tokens)

        q_in  = x if s_q == 1 else F.avg_pool3d(x, kernel_size=s_q, stride=s_q, ceil_mode=True)
        kv_in = x if s_kv == 1 else F.avg_pool3d(x, kernel_size=s_kv, stride=s_kv, ceil_mode=True)

        _, _, Dq, Hq, Wq = q_in.shape
        _, _, Dk, Hk, Wk = kv_in.shape
        Nq = Dq * Hq * Wq
        Mk = Dk * Hk * Wk

        if Nq * Mk > self.max_attn_elems:
            return x + self.se_scale * (x * w)

        theta = self.theta(q_in).view(B, -1, Nq).transpose(1, 2)  # (B, Nq, C')
        phi   = self.phi(kv_in).view(B, -1, Mk)                   # (B, C', Mk)
        g     = self.g(kv_in).view(B, -1, Mk).transpose(1, 2)     # (B, Mk, C')

        attn = torch.matmul(theta, phi) * self._scale
        attn = F.softmax(attn, dim=-1)
        y    = torch.matmul(attn, g)                               # (B, Nq, C')
        y    = y.transpose(1, 2).contiguous().view(B, -1, Dq, Hq, Wq)

        if (Dq, Hq, Wq) != (D, H, W):
            y = F.interpolate(y, size=(D, H, W), mode="trilinear", align_corners=False)

        y = self.out(y)
        return x + self.res_scale * y + self.se_scale * (x * w)

class UpResizeConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, target_size):
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="nearest")
        x = self.conv(x)
        if x.shape[2] * x.shape[3] * x.shape[4] > 1:
            x = self.norm(x)
        x = self.act(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_channels=64):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        enc_filters = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
            base_channels * 8,
            base_channels * 8,
            base_channels * 8,
        ]
        enc_filters = [min(f, 512) for f in enc_filters]

        enc_kernels = [(4, 4, 4)] * 8
        enc_strides = [
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
        ]
        enc_pads = [(1, 1, 1)] * 8

        self.enc_convs = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.enc_attn_light = nn.ModuleList()

        prev_ch = in_channels
        for i, f in enumerate(enc_filters):
            ks = enc_kernels[i]
            st = enc_strides[i]
            pd = enc_pads[i]
            self.enc_convs.append(nn.Conv3d(prev_ch, f, kernel_size=ks, stride=st, padding=pd, bias=False))
            self.enc_blocks.append(ResBlock3D(f, f))
            self.enc_attn.append(SelfAttn3D(f))
            self.enc_attn_light.append(SelfAttentionBlock(f))
            prev_ch = f

        bottleneck_idx = len(self.enc_blocks) - 1
        bn_ch = enc_filters[-1]
        gn_groups = 32 if bn_ch % 32 == 0 else 16 if bn_ch % 16 == 0 else 8
        self.enc_blocks[bottleneck_idx].norm1 = nn.GroupNorm(num_groups=gn_groups, num_channels=bn_ch)
        self.enc_blocks[bottleneck_idx].norm2 = nn.GroupNorm(num_groups=gn_groups, num_channels=bn_ch)

        self.bottleneck_block1 = ResBlock3D(bn_ch, bn_ch)
        self.bottleneck_attn   = SelfAttn3D(bn_ch)
        self.bottleneck_block2 = ResBlock3D(bn_ch, bn_ch)
        for blk in (self.bottleneck_block1, self.bottleneck_block2):
            blk.norm1 = nn.GroupNorm(num_groups=gn_groups, num_channels=bn_ch)
            blk.norm2 = nn.GroupNorm(num_groups=gn_groups, num_channels=bn_ch)

        self.use_bottleneck_block2 = False

        self.dec_ups = nn.ModuleList()
        self.dec_attn_gates = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.dec_attn_light = nn.ModuleList()

        for i in range(len(enc_filters) - 2, -1, -1):
            in_ch  = enc_filters[i + 1]
            out_ch = enc_filters[i]
            self.dec_ups.append(UpResizeConv3D(in_ch, out_ch))
            self.dec_attn_gates.append(AttnGate3D(skip_channels=enc_filters[i], gating_channels=out_ch))
            self.dec_blocks.append(ResBlock3D(in_channels=out_ch + enc_filters[i], out_channels=out_ch))
            self.dec_attn.append(SelfAttn3D(out_ch))
            self.dec_attn_light.append(SelfAttentionBlock(out_ch))

        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.out_act = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, (nn.InstanceNorm3d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1, 1).expand(-1, -1, x.size(2), x.size(3), x.size(4))
        x_in = torch.cat([x, c], dim=1)

        if x_in.size(1) != self.in_channels:
            raise RuntimeError(
                f"Channel mismatch: got x+c = {x_in.size(1)} channels, "
                f"but Generator(in_channels={self.in_channels})."
            )

        # Encoder
        enc_feats = []
        out = x_in
        for i in range(len(self.enc_convs)):
            out = self.enc_convs[i](out)
            if out.shape[2] * out.shape[3] * out.shape[4] > 1:
                out = F.instance_norm(out)

            out = F.leaky_relu(out, 0.2, inplace=True)
            out = self.enc_blocks[i](out)

            if i >= 3 and i < len(self.enc_convs) - 1:
                out = self.enc_attn[i](out)
            else:
                if i < len(self.enc_convs) - 1:
                    out = self.enc_attn_light[i](out)

            enc_feats.append(out)
        dec = enc_feats[-1]
        dec = self.bottleneck_block1(dec)
        dec = self.bottleneck_attn(dec)
        if self.use_bottleneck_block2:
            dec = self.bottleneck_block2(dec)

        skip_levels = len(enc_feats) - 1
        for idx in range(skip_levels):
            i = skip_levels - 1 - idx
            skip_feat = enc_feats[i]
            target_size = skip_feat.shape[2:]

            dec = self.dec_ups[idx](dec, target_size=target_size)

            attn_skip = self.dec_attn_gates[idx](skip_feat, dec)
            fusion = torch.cat((attn_skip, dec), dim=1)
            dec = self.dec_blocks[idx](fusion)

            if i >= 3:
                dec = self.dec_attn[idx](dec)
            else:
                dec = self.dec_attn_light[idx](dec)

        if dec.shape[2:] != x.shape[2:]:
            dec = F.interpolate(dec, size=x.shape[2:], mode="trilinear", align_corners=False)
        dec = self.final_conv(dec)
        dec = self.out_act(dec)
        return dec

