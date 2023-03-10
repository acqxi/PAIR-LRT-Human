import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResBlockDown


class Discriminator(nn.Module):

    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg

        self.c1 = ResBlockDown(3, cfg['D_CONV_CH'], pre_activation=False, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.c2 = ResBlockDown(cfg['D_CONV_CH'], cfg['D_CONV_CH'] * 2, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.c3 = ResBlockDown(cfg['D_CONV_CH'] * 2, cfg['D_CONV_CH'] * 4, neg_slope=cfg['RELU_NEG_SLOPE'])

        self.label_dense = nn.Linear(
            cfg['D_CONV_CH'] * 4 * (1 if cfg['D_USE_GSP'] else cfg['IMG_SHAPE_Y'] // 8 * cfg['IMG_SHAPE_X'] // 8), 1)

        self.thermal_rec = nn.Sequential(
            ResBlockDown(cfg['D_CONV_CH'] * 4, cfg['D_CONV_THERMAL_CH'], down_size=1, neg_slope=cfg['RELU_NEG_SLOPE']),
            nn.LeakyReLU(negative_slope=cfg['RELU_NEG_SLOPE']), nn.Conv2d(cfg['D_CONV_THERMAL_CH'], 1, 3, padding=1), nn.Tanh())

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)

        feature = F.leaky_relu(c3, negative_slope=self.cfg['RELU_NEG_SLOPE'])
        if self.cfg['D_USE_GSP']:
            feature = torch.sum(feature, dim=[2, 3])
        else:
            feature = torch.flatten(feature, start_dim=1)
        label = self.label_dense(feature)

        thermal_rec = self.thermal_rec(c3)

        return label, thermal_rec


if __name__ == "__main__":

    # Usage
    cfg = {
        'IMG_SHAPE_Y': 80,
        'IMG_SHAPE_X': 128,
        'D_CONV_CH': 64,
        'D_CONV_THERMAL_CH': 32,
        'RELU_NEG_SLOPE': 0.2,
        'D_USE_GSP': False  # Global Sum Pooling
    }

    discriminator = Discriminator(cfg)
    img_input = torch.randn(1, 3, cfg['IMG_SHAPE_Y'], cfg['IMG_SHAPE_X'])
    label, thermal_rec = discriminator(img_input)

    print(label.shape, thermal_rec.shape)
