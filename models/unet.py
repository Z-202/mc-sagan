import torch
import torch.nn as nn

class Unet(nn.Module):

    def __init__(self, in_dim=4, conv_dim=32, out_dim=1):
        super(Unet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_dim, conv_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(conv_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(conv_dim, conv_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(conv_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(conv_dim * 2, conv_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(conv_dim * 4),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(conv_dim * 4, conv_dim * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(conv_dim * 8),
            nn.ReLU(inplace=True),
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(conv_dim * 8, conv_dim * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(conv_dim * 8),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(conv_dim * (8 + 4), conv_dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(conv_dim * 4),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(conv_dim * (4 + 2), conv_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(conv_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(conv_dim * (2 + 1), out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        out = self.deconv1(x4)
        x3  = torch.cat([x3, out], dim=1)

        out = self.deconv2(x3)
        x2  = torch.cat([x2, out], dim=1)

        out = self.deconv3(x2)
        x1  = torch.cat([x1, out], dim=1)

        out = self.deconv4(x1)
        return out
