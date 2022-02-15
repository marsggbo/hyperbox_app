import torch
import torch.nn as nn
import math

from hyperbox.mutables.spaces import OperationSpace
from hyperbox.utils.utils import load_json

from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox_app.medmnist.networks.aug_v2 import DataAugmentation
from hyperbox_app.medmnist.networks.mobile3d_ops import *
from hyperbox_app.medmnist.networks.mobile_utils import *


__all__ = [
    'Mobile3DNet',
    'DAMobile3DNet'
]


class Mobile3DNet(BaseNASNetwork):
    DEFAULT_OPS = [
        '3x3_MBConv3',
        '3x3_MBConv4',
        '3x3_MBConv6',
        '5x5_MBConv3',
        '5x5_MBConv4',
        '7x7_MBConv3',
        '7x7_MBConv4',
        'Identity'
    ]
    def __init__(
        self, in_channels=3,
        input_channel=None,
        first_stride=1,
        width_stages=[24,40,80,96,192,320],
        n_cell_stages=[4,4,4,4,4,1],
        stride_stages=[2,2,2,1,2,1],
        width_mult=1, num_classes=1000,
        dropout_rate=0, bn_param=(0.1, 1e-3),
        last_channel=1280,
        candidate_ops=None,
        mask=None
    ):
        """
        Parameters
            ----------
            width_stages: str
                width (output channels) of each cell stage in the block
            n_cell_stages: str
                number of cells in each cell stage
            stride_strages: str
                stride of each cell stage in the block
            width_mult : int
                the scale factor of width
        """
        super(Mobile3DNet, self).__init__(mask)
        if candidate_ops is not None:
            self.candidate_ops = candidate_ops
        else:
            self.candidate_ops = self.DEFAULT_OPS
        if input_channel is None:
            input_channel = make_divisible(16 * width_mult, 8)
        first_cell_width = make_divisible(24 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)
        # first conv
        self.first_conv = ConvLayer(in_channels, input_channel, kernel_size=3, stride=first_stride, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # self.first_conv = OPS['3x3_MBConv3'](in_channels, input_channel, first_stride)

        # first block
        first_block = OPS['3x3_MBConv3'](input_channel, first_cell_width, 1)

        input_channel = first_cell_width
        blocks = [first_block]
        stage_cnt = 0
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                calibrate_op = None
                if i == 0:
                    stride = 1
                    if stage_cnt <= 1:
                        calibrate_op = CalibrationLayer(input_channel, width, stride=s)
                    else:
                        calibrate_op = OperationSpace(
                            [CalibrationLayer(input_channel, width, stride=1),
                            CalibrationLayer(input_channel, width, stride=2),
                            ], mask=self.mask, return_mask=False, key="s{}_calib".format(stage_cnt)
                        )
                else:
                    stride = 1
                    # calibrate_op = CalibrationLayer(input_channel, width, stride=1)
                if calibrate_op is not None: blocks.append(calibrate_op)
                op_candidates = [
                    OPS[key](width, width, 1) for key in self.candidate_ops
                ]
                conv_op = OperationSpace(op_candidates, mask=self.mask, return_mask=False, key="s{}_c{}".format(stage_cnt, i))
                # shortcut
                if stride == 1:
                    # if not first cell
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut, op_candidates)
                blocks.append(inverted_residual_block)
                input_channel = width
            stage_cnt += 1
        self.blocks = nn.ModuleList(blocks)

        # feature mix layer
        # last_channel = input_channel
        last_channel = make_devisible(self.last_channel * width_mult, 8) if width_mult > 1.0 else self.last_channel
        self.feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1,
            use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        ) # disable activation, otherwise the final predictions will be the same class for all inputs

        self.global_avg_pooling = nn.AdaptiveAvgPool3d(1)
        self.classifier = LinearLayer(
            last_channel, num_classes, dropout_rate=dropout_rate, bias=False)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
        self.init_model()

    def forward(self, x, **kwargs):
        gamma = 0.8 # 0 equals to training mode
        x = self.first_conv(x)
        try:
            for idx, block in enumerate(self.blocks):
                x = block(x)
        except Exception as e:
            print(str(e), idx)
            print(f"{idx} block {x.shape}")
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        # if not self.training:
        #     w = self.classifier.linear.weight.detach()
        #     w = w / (w.norm(dim=1)**gamma).view(w.shape[0], -1)
        #     x = nn.functional.linear(x, w)
        # else:
        #     x = self.classifier(x)
        x = self.classifier(x)
        return x

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def init_model(self, model_init='he_fout', init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def reset_first_block(self, in_channels):
        self.first_conv = ConvLayer(
            in_channels, input_channel, kernel_size=3, stride=2,
            use_bn=True, act_func='relu6', ops_order='weight_bn_act')

    @property
    def arch(self):
        arch = ''
        for module in self.blocks:
            if isinstance(module, MobileInvertedResidualBlock):
                index = module.mobile_inverted_conv.mask.cpu().detach().numpy().argmax()
                arch +=f'{index}-'
        return arch


class DAMobile3DNet(BaseNASNetwork):
    def __init__(
        self,
        in_channels=3, 
        input_channel=None,
        first_stride=1, width_stages=[24,40,80,96,192,320],
        n_cell_stages=[4,4,4,4,4,1],
        stride_stages=[2,2,2,1,2,1],
        width_mult=1, num_classes=1000,
        dropout_rate=0, bn_param=(0.1, 1e-3),
        last_channel=1280,
        candidate_ops=None,

        rotate_degree=30, crop_size=[(32,128,128), (32,256,256)],
        affine_degree=[(0,0,5.)], affine_scale=(1.1, 1.5), affine_shears=0,
        brightness=[0.3, 0.5], contrast=[0.3,0.8],
        blur_ks=[(3,3), (2,2)],
        invert_val=[0.25, 0.5, 0.75, 1],
        noise_mean=0.1, noise_std=0.05,
        erase_scale=[(0.02, 0.1), (0.1, 0.2)], erase_ratio=[(0.3, 3.3)],
        mean=0.5, std=0.5, aug_keys=None, ignore_keys=['invert', 'rcrop'],
        mask=None
    ):
        super(DAMobile3DNet, self).__init__(mask)
        self.network = Mobile3DNet(
            in_channels, input_channel, first_stride, width_stages, n_cell_stages, stride_stages, width_mult,
            num_classes, dropout_rate, bn_param, last_channel, candidate_ops, mask)
        self.augmentation = DataAugmentation(
            rotate_degree, crop_size, affine_degree, affine_scale, affine_shears,
            brightness, contrast, blur_ks, invert_val, noise_mean, noise_std,
            erase_scale, erase_ratio, mean, std, aug_keys, ignore_keys, mask)

    def forward(self, x, to_aug=False):
        x = self.augmentation(x, to_aug)
        self.aug_imgs = x.detach()
        x = self.network(x)
        return x

if __name__ == '__main__':
    from hyperbox.mutator import DartsMutator, RandomMutator, OnehotMutator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = DAMobile3DNet(1, num_classes=10).to(device)
    # net = Mobile3DNet(1, num_classes=10).to(device)
    dm = OnehotMutator(net)
    for i in range(10):
        dm.reset()
        # if i > 5:
        #     net = net.eval()
        x = torch.rand(8,1,32,64,64).to(device)
        y = net(x)
        print(y.argmax(-1))
