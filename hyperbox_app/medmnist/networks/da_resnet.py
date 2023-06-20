import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter

from hyperbox.mutables.spaces import OperationSpace
from hyperbox.utils.utils import load_json

from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox_app.medmnist.networks.aug_v2 import DataAugmentation


__all__ = [
    'DAResNet'
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                # nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        # self.bn3 = nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                # nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = F.adaptive_avg_pool3d(out, output_size=4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_channels, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)


def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)


class DAResNet(BaseNASNetwork):
    def __init__(
        self,
        in_channels=3, 
        num_classes=10,
        resnet_version='18', # or '50'
        converter='2.5d', # or '3d', 'ACS'
        rotate_degree=30, crop_size=[(28,28,28)],
        affine_degree=[0, 3], affine_scale=[(0.8, 1), (1.1, 1.6)], affine_shears=[0, 2],
        brightness=[0.3, 0.5], contrast=[0.3,0.8],
        blur_ks=[(3,3), (2,2)],
        invert_val=[0.25, 0.5, 0.75, 1],
        noise_mean=0.1, noise_std=0.05,
        erase_scale=[(0.02, 0.1), (0.1, 0.2)], erase_ratio=[(0.3, 3.3)],
        mean=0.5, std=0.5, aug_keys=None, ignore_keys=['invert', 'rcrop'],
        mask=None
    ):
        super(DAResNet, self).__init__(mask)
        if resnet_version == '18':
            network = ResNet18(in_channels, num_classes)
        else:
            assert resnet_version == '50', 'only support resnet18 or resnet50'
            network = ResNet50(in_channels, num_classes)
        if converter == '2.5d':
            self.network = Conv2_5dConverter(network).model
        elif converter == '3d':
            self.network = Conv3dConverter(network).model
        else:
            assert converter == 'ACS', 'only support 2.5d, 3d, and ACS converter'
            self.network = ACSConverter(network).model
        self.augmentation = DataAugmentation(
            rotate_degree, crop_size, affine_degree, affine_scale, affine_shears,
            brightness, contrast, blur_ks, invert_val, noise_mean, noise_std, erase_scale, erase_ratio,
            mean, std, aug_keys, ignore_keys, mask)

    def forward(self, x, to_aug=False):
        x = self.augmentation(x, to_aug)
        x = self.network(x)
        return x

if __name__ == '__main__':
    from hyperbox.mutator import DartsMutator, RandomMutator, OnehotMutator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = DAResNet(1, num_classes=10).to(device)
    dm = OnehotMutator(net)
    for i in range(10):
        dm.reset()
        if i > 5:
            net = net.eval()
        x = torch.rand(10,1,28,28,28).to(device)
        y = net(x, True)
        print(y.argmax(-1))

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