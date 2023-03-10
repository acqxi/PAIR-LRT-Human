import torch
import torch.nn as nn

from .layers import ResBlockDown


class InverseEncoder(nn.Module):

    def __init__(self, cfg):
        super(InverseEncoder, self).__init__()
        self.cfg = cfg

        self.resblock1 = ResBlockDown(3, cfg['E_CONV_CH'], use_bn=True, pre_activation=False, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.resblock2 = ResBlockDown(cfg['E_CONV_CH'], cfg['E_CONV_CH'] * 2, use_bn=True, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.resblock3 = ResBlockDown(cfg['E_CONV_CH'] * 2, cfg['E_CONV_CH'] * 4, use_bn=True, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.resblock4 = ResBlockDown(cfg['E_CONV_CH'] * 4, cfg['E_CONV_CH'] * 8, use_bn=True, neg_slope=cfg['RELU_NEG_SLOPE'])

        self.bn = nn.BatchNorm2d(cfg['E_CONV_CH'] * 8)
        self.relu = nn.LeakyReLU(negative_slope=cfg['RELU_NEG_SLOPE'])

        if cfg['E_USE_GSP']:
            self.pool = lambda x: torch.sum(x, dim=[2, 3])
        else:
            self.pool = nn.Flatten()

        self.fc = nn.Linear(cfg['E_CONV_CH'] * 8 *
                            (1 if cfg['E_USE_GSP'] else cfg['IMG_SHAPE_Y'] // 16 * cfg['IMG_SHAPE_X'] // 16),
                            cfg['G_Z_DIM'],
                            bias=False)
        self.bn_out = nn.BatchNorm1d(cfg['G_Z_DIM'])

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = self.bn(x)
        x = self.relu(x)

        x = self.pool(x)

        z = self.fc(x)
        z = self.bn_out(z)

        return z


if __name__ == '__main__':
    # Usage
    cfg = {
        'IMG_SHAPE_Y': 80,
        'IMG_SHAPE_X': 128,
        'E_CONV_CH': 64,
        'G_Z_DIM': 128,
        'RELU_NEG_SLOPE': 0.2,
        'E_USE_GSP': False  # Global Sum Pooling
    }

    encoder = InverseEncoder(cfg)
    img_input = torch.randn(1, 3, cfg['IMG_SHAPE_Y'], cfg['IMG_SHAPE_X'])
    z = encoder(img_input)

    print(z.shape)
