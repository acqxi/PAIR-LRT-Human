import torch.nn as nn
import torch.nn.functional as F


class ResBlockDown(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 down_size=2,
                 use_bn=True,
                 pre_activation=True,
                 neg_slope=0.0):
        super(ResBlockDown, self).__init__()
        self.down_size = down_size
        self.use_bn = use_bn
        self.pre_activation = pre_activation
        self.neg_slope = neg_slope

        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=not use_bn)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size[0] // 2, bias=not use_bn)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size[0] // 2, bias=not use_bn)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AvgPool2d(down_size) if down_size > 1 else nn.Identity()

    def forward(self, x):
        skip = self.skip_conv(x)
        skip = self.avg_pool(skip)

        if self.use_bn:
            x = self.bn1(x)
        if self.pre_activation:
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
        x = self.conv1(x)

        if self.use_bn:
            x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=self.neg_slope)
        x = self.conv2(x)

        x = self.avg_pool(x)
        return x + skip


class ResBlockUp(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 up_size=2,
                 use_bn=True,
                 pre_activation=True,
                 neg_slope=0.0):
        super(ResBlockUp, self).__init__()
        self.up_size = up_size
        self.use_bn = use_bn
        self.pre_activation = pre_activation
        self.neg_slope = neg_slope

        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=not use_bn)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size[0] // 2, bias=not use_bn)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size[0] // 2, bias=not use_bn)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        skip = F.interpolate(x, scale_factor=self.up_size, mode='nearest') if self.up_size > 1 else x
        skip = self.skip_conv(skip)

        if self.use_bn:
            x = self.bn1(x)
        if self.pre_activation:
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
        x = F.interpolate(x, scale_factor=self.up_size, mode='nearest') if self.up_size > 1 else x
        x = self.conv1(x)

        if self.use_bn:
            x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=self.neg_slope)
        x = self.conv2(x)

        return x + skip


class SpadeSR(nn.Module):

    def __init__(self, x_ch, spade_filters, kernel_size=(3, 3)):
        super(SpadeSR, self).__init__()
        self.bn = nn.BatchNorm2d(x_ch, affine=False)
        self.conv = nn.Conv2d(spade_filters, spade_filters, kernel_size, padding=kernel_size[0] // 2)
        self.conv_gamma = nn.Conv2d(spade_filters, x_ch, kernel_size, padding=kernel_size[0] // 2)
        self.conv_beta = nn.Conv2d(spade_filters, x_ch, kernel_size, padding=kernel_size[0] // 2)

    def forward(self, x, m):
        normalized = self.bn(x)
        m = F.relu(self.conv(m))
        gamma = self.conv_gamma(m)
        beta = self.conv_beta(m)
        return normalized * (1 + gamma) + beta


class ResBlockUpSpadeSR(nn.Module):

    def __init__(self, in_channels, out_channels, spade_filters, kernel_size=(3, 3), up_size=2, neg_slope=0.0):
        super(ResBlockUpSpadeSR, self).__init__()
        self.up_size = up_size
        self.neg_slope = neg_slope

        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size[0] // 2, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size[0] // 2, bias=False)

        self.spade1 = SpadeSR(in_channels, spade_filters)
        self.spade2 = SpadeSR(out_channels, spade_filters)

    def forward(self, x, m1, m2):
        skip = F.interpolate(x, scale_factor=self.up_size, mode='nearest') if self.up_size > 1 else x
        skip = self.skip_conv(skip)

        x = self.spade1(x, m1)
        x = F.leaky_relu(x, negative_slope=self.neg_slope)
        x = F.interpolate(x, scale_factor=self.up_size, mode='nearest') if self.up_size > 1 else x
        x = self.conv1(x)

        x = self.spade2(x, m2)
        x = F.leaky_relu(x, negative_slope=self.neg_slope)
        x = self.conv2(x)

        return x + skip
