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
        '5x5_MBConv3',
        '7x7_MBConv3',
        'Identity'
    ]
    def __init__(
        self, in_channels=3,
        first_stride=1,
        width_stages=[16,24,40,80,96,128,160,320],
        n_cell_stages=[2,3,3,3,3,3,3,2],
        stride_stages=[2,2,1,1,2,1,1,1],
        width_mult=1, num_classes=1000,
        dropout_rate=0, bn_param=(0.1, 1e-3),
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
        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)
        # first conv
        self.first_conv = ConvLayer(
            in_channels, input_channel, kernel_size=3, stride=first_stride,
            use_bn=True, act_func='relu6', ops_order='weight_bn_act')

        # first block
        first_block = OPS['3x3_MBConv1'](input_channel, first_cell_width, 1)

        input_channel = first_cell_width
        blocks = [first_block]
        stage_cnt = 0
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                layer_op = []
                if i == 0:
                    # s=2: stride is 2
                    # s=1: no stride_op
                    # s=0: needs to search stride
                    if s == 2:
                        stride_op = OPS['3x3_MBConv3SE'](input_channel, width, 2)
                    elif s == 0:
                        stride_op = OperationSpace(
                            candidates=[
                                OPS['3x3_MBConv3SE'](input_channel, width, 1),
                                OPS['3x3_MBConv3SE'](input_channel, width, 2)
                            ],
                            mask=self.mask, return_mask=False, key="stride_op{}".format(stage_cnt)
                        )
                    input_channel = width
                    layer_op.append(stride_op)

                op_candidates = [
                    OPS[key](input_channel, width, 1) for key in self.candidate_ops
                ]
                conv_op = OperationSpace(op_candidates, mask=self.mask, return_mask=True, key="s{}_c{}".format(stage_cnt, i))
                shortcut = IdentityLayer(input_channel, input_channel)
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut, op_candidates)
                layer_op.append(inverted_residual_block)

                layer_op = nn.Sequential(*layer_op)
                blocks.append(layer_op)
                input_channel = width
            stage_cnt += 1
        self.blocks = nn.ModuleList(blocks)

        # feature mix layer
        # last_channel = input_channel
        last_channel = make_devisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1,
            use_bn=True, act_func=None, ops_order='weight_bn_act'
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
        for idx, block in enumerate(self.blocks):
            x = block(x)
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
        in_channels=3, first_stride=1, width_stages=[24,40,80,96,192,320],
        n_cell_stages=[4,4,4,4,4,1],
        stride_stages=[2,2,2,1,2,1],
        width_mult=1, num_classes=1000,
        dropout_rate=0, bn_param=(0.1, 1e-3),
        candidate_ops=None,

        rotate_degree=30, crop_size=[(32,128,128), (32,256,256)],
        affine_degree=0, affine_scale=(1.1, 1.5), affine_shears=20,
        mean=0.5, std=0.5,
        mask=None
    ):
        super(DAMobile3DNet, self).__init__(mask)
        self.network = Mobile3DNet(
            in_channels, first_stride, width_stages, n_cell_stages, stride_stages, width_mult,
            num_classes, dropout_rate, bn_param, candidate_ops, mask)
        self.augmentation = DataAugmentation(
            rotate_degree, crop_size, affine_degree, affine_scale, affine_shears, mean, std, mask)

    def forward(self, x, to_aug=False):
        x = self.augmentation(x, to_aug)
        x = self.network(x)
        return x

if __name__ == '__main__':
    from hyperbox.mutator import DartsMutator, RandomMutator, OnehotMutator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = DAMobile3DNet(1, num_classes=10, crop_size=[[8,360,360], [12,512,512]], stride_stages=[2,2,1,2,1,1], first_stride=2).to(device)
    # net = Mobile3DNet(1, n_cell_stages=[2,2,2,2,2,2], num_classes=10).to(device)
    dm = OnehotMutator(net)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for i in range(10):
        dm.reset()
        print('\nsampling...')
        opt.zero_grad()
        # if i > 5:
        #     net = net.eval()
        x = torch.rand(8,1,32,512,512).to(device)
        y = net(x)
        print(y.argmax(-1))
        z = y.sum()
        z.backward()
        opt.step()

    # from omegaconf import OmegaConf
    # from hyperbox.networks.utils import extract_net_from_ckpt
    # cfg = OmegaConf.load('/home/comp/18481086/code/hyperbox_app/logs/runs/gdas_fracture3d_gpu1_batchbalance/2022-01-06_15-33-58/.hydra_only_test/config.yaml')
    # netcfg = cfg.model.network_cfg
    # netcfg.pop('_target_')
    # mask = '/home/comp/18481086/code/hyperbox_app/logs/runs/gdas_fracture3d_gpu1_batchbalance/2022-01-06_15-33-58/mask_json/mask_epoch_3.json'
    # # netcfg.mask=mask
    # net = DAMobile3DNet(**netcfg)
    # ckpt = '/home/comp/18481086/code/hyperbox_app/logs/runs/gdas_fracture3d_gpu1_batchbalance/2022-01-06_15-33-58/checkpoints/last.ckpt'
    # weight_supernet = extract_net_from_ckpt(ckpt)
    # print('extract_net_from_ckpt')
    # net.load_from_supernet(weight_supernet)
    # print('load_from_supernet')