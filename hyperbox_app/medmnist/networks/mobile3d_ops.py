from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbox_app.medmnist.networks.mobile_utils import get_same_padding, build_activation


OPS = {
    'Identity': lambda in_C, out_C, stride: IdentityLayer(in_C, out_C, ops_order='weight_bn_act'),
    'Zero': lambda in_C, out_C, stride: ZeroLayer(stride=stride),
    '3x3_MBConv1': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 1),
    '3x3_MBConv2': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 2),
    '3x3_MBConv3': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 3),
    '3x3_MBConv4': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 4),
    '3x3_MBConv5': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 5),
    '3x3_MBConv6': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 6),
    '5x5_MBConv1': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 1),
    '5x5_MBConv2': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 2),
    '5x5_MBConv3': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 3),
    '5x5_MBConv4': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 4),
    '5x5_MBConv5': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 5),
    '5x5_MBConv6': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 6),
    '7x7_MBConv1': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 1),
    '7x7_MBConv2': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 2),
    '7x7_MBConv3': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 3),
    '7x7_MBConv4': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 4),
    '7x7_MBConv5': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 5),
    '7x7_MBConv6': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 6),
    '3x3_MBConv3SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 3, use_se=True),
    '3x3_MBConv4SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 4, use_se=True),
    '3x3_MBConv5SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 5, use_se=True),
    '3x3_MBConv6SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 3, stride, 6, use_se=True),
    '5x5_MBConv3SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 3, use_se=True),
    '5x5_MBConv4SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 4, use_se=True),
    '5x5_MBConv5SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 5, use_se=True),
    '5x5_MBConv6SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 5, stride, 6, use_se=True),
    '7x7_MBConv3SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 3, use_se=True),
    '7x7_MBConv4SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 4, use_se=True),
    '7x7_MBConv5SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 5, use_se=True),
    '7x7_MBConv6SE': lambda in_C, out_C, stride: MBInvertedConvLayer(in_C, out_C, 7, stride, 6, use_se=True)
}


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hsigmoid()'

class SEModule3D(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super(SEModule3D, self).__init__()

        self.channel = channel
        self.reduction = SEModule3D.REDUCTION if reduction is None else reduction

        num_mid = make_divisible(self.channel // self.reduction, divisor=8)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv3d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', nn.Conv3d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))

    def forward(self, x):
        y = x.mean(4, keepdim=True).mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y

    def __repr__(self):
        return 'SE(channel=%d, reduction=%d)' % (self.channel, self.reduction)


class MobileInvertedResidualBlock(nn.Module):
    
    def __init__(self, mobile_inverted_conv, shortcut, op_candidates_list=None):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        self.op_candidates_list = op_candidates_list

    def forward(self, x):
        out = self.mobile_inverted_conv(x)
        idx = getattr(self.mobile_inverted_conv, 'mask', None)
        # TODO: unify idx format
        if idx is not None:
            idx = torch.argmax(idx.float())
        if idx is not None and self.op_candidates_list[idx].is_zero_layer():
            res = x
        elif self.shortcut is None:
            res = out
        else:
            conv_x = out
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res


class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, depth, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, depth, height, width)
        # noinspection PyUnresolvedReferences
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, depth, height, width)
        return x


class Base3DLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(Base3DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm3d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm3d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout3d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(Base3DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv3d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)

        return weight_dict


class IdentityLayer(Base3DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        return None


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_features)
            else:
                modules['bn'] = nn.BatchNorm1d(out_features)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # linear
        modules['weight'] = {'linear': nn.Linear(self.in_features, self.out_features, self.bias)}

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @staticmethod
    def is_zero_layer():
        return False


class MBInvertedConvLayer(nn.Module):
    """
    This layer is introduced in section 4.2 in the paper https://arxiv.org/pdf/1812.00332.pdf
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, use_se=False):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.use_se = use_se

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv3d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm3d(feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm3d(feature_dim)),
            ('act', nn.ReLU6(inplace=True)),
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', SEModule3D(feature_dim))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm3d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @staticmethod
    def is_zero_layer():
        return False


class CalibrationLayer(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(CalibrationLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroLayer(nn.Module):
    
    def __init__(self, stride=None):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        '''n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding'''
        if self.stride == 1:
            return x.mul(0.)
        if len(x.shape)==4:
            return x[:,:,::self.stride,::self.stride].mul(0.)
        elif len(x.shape)==5:
            return x[:,:,::self.stride,::self.stride,::self.stride].mul(0.)

    @staticmethod
    def is_zero_layer():
        return True

