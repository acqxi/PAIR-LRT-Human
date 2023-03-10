import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResBlockUp, ResBlockUpSpadeSR, SpadeSR


class Generator(nn.Module):

    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg

        # Thermal pathway
        self.thermal_c1 = ResBlockUp(1,
                                     cfg['G_CONV_THERMAL_CH'],
                                     pre_activation=False,
                                     use_bn=False,
                                     neg_slope=cfg['RELU_NEG_SLOPE'],
                                     up_size=1)
        self.thermal_c2 = ResBlockUp(cfg['G_CONV_THERMAL_CH'],
                                     cfg['G_CONV_THERMAL_CH'],
                                     neg_slope=cfg['RELU_NEG_SLOPE'],
                                     use_bn=False)
        self.thermal_c3 = ResBlockUp(cfg['G_CONV_THERMAL_CH'],
                                     cfg['G_CONV_THERMAL_CH'],
                                     neg_slope=cfg['RELU_NEG_SLOPE'],
                                     use_bn=False)
        self.thermal_c4 = ResBlockUp(cfg['G_CONV_THERMAL_CH'],
                                     cfg['G_CONV_THERMAL_CH'],
                                     neg_slope=cfg['RELU_NEG_SLOPE'],
                                     use_bn=False)

        # Main pathway
        self.fc = nn.Linear(cfg['G_Z_DIM'], 5 * 8 * cfg['G_CONV_CH'] * 4)
        self.resblock1 = ResBlockUpSpadeSR(cfg['G_CONV_CH'] * 4,
                                           cfg['G_CONV_CH'] * 4,
                                           cfg['G_CONV_THERMAL_CH'],
                                           neg_slope=cfg['RELU_NEG_SLOPE'])
        self.resblock2 = ResBlockUpSpadeSR(cfg['G_CONV_CH'] * 4,
                                           cfg['G_CONV_CH'] * 2,
                                           cfg['G_CONV_THERMAL_CH'],
                                           neg_slope=cfg['RELU_NEG_SLOPE'])
        self.resblock3 = ResBlockUpSpadeSR(cfg['G_CONV_CH'] * 2,
                                           cfg['G_CONV_CH'],
                                           cfg['G_CONV_THERMAL_CH'],
                                           neg_slope=cfg['RELU_NEG_SLOPE'])

        self.spade_final = SpadeSR(cfg['G_CONV_CH'], cfg['G_CONV_THERMAL_CH'])
        self.conv_out = nn.Conv2d(cfg['G_CONV_CH'], 3, 3, padding=1)

    def forward(self, thermal_input, noise):
        # Thermal pathway
        t1 = self.thermal_c1(thermal_input)
        t2 = self.thermal_c2(t1)
        t3 = self.thermal_c3(t2)
        t4 = self.thermal_c4(t3)

        # Main pathway
        x = self.fc(noise).view(-1, self.cfg['G_CONV_CH'] * 4, 5, 8)
        x = self.resblock1(x, F.leaky_relu(t1, negative_slope=self.cfg['RELU_NEG_SLOPE']),
                           F.leaky_relu(t2, negative_slope=self.cfg['RELU_NEG_SLOPE']))
        x = self.resblock2(x, F.leaky_relu(t2, negative_slope=self.cfg['RELU_NEG_SLOPE']),
                           F.leaky_relu(t3, negative_slope=self.cfg['RELU_NEG_SLOPE']))
        x = self.resblock3(x, F.leaky_relu(t3, negative_slope=self.cfg['RELU_NEG_SLOPE']),
                           F.leaky_relu(t4, negative_slope=self.cfg['RELU_NEG_SLOPE']))

        x = self.spade_final(x, F.leaky_relu(t4, negative_slope=self.cfg['RELU_NEG_SLOPE']))
        x = F.leaky_relu(x, negative_slope=self.cfg['RELU_NEG_SLOPE'])
        x = self.conv_out(x)
        return torch.tanh(x)


if "__main__" == __name__:
    # Usage
    cfg = {'G_CONV_THERMAL_CH': 64, 'G_CONV_CH': 64, 'G_Z_DIM': 128, 'RELU_NEG_SLOPE': 0.2}

    generator = Generator(cfg)
    thermal_input = torch.randn(1, 1, 5, 8)
    noise = torch.randn(1, cfg['G_Z_DIM'])
    output = generator(thermal_input, noise)

    print(output.shape)  # torch.Size([1, 3, 40, 64])

    import matplotlib.pyplot as plt

    plt.imshow(output[0].detach().numpy().transpose((1, 2, 0)) / 2 + 0.5)
