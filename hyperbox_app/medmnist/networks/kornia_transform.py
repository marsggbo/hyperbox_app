import os
from typing import Union

import torch
import torch.nn as nn
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import *
from omegaconf.listconfig import ListConfig

from hyperbox.mutables.spaces import OperationSpace
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox_app.medmnist.networks.kornia_aug import (RandomBoxBlur3d,
                                                       RandomErasing3d,
                                                       RandomGaussianNoise3d,
                                                       RandomInvert3d,
                                                       RandomSharpness3d,
                                                       RandomResizedCrop3d)

__all__ = [
    'DataAugmentation',
    'DAOperation3D'
]


def prob_list_gen(func, num_probs=4, probs: list=None, *args, **kwargs):
    if probs is not None:
        return [func(p=p, *args, **kwargs) for p in probs]
    else:
        return [func(p=p, *args, **kwargs) for p in [i*0.25 for i in range(num_probs)]]

def DAOperation3D(
    affine_degree=30, affine_scale=(1.1, 1.5), affine_shears=20,
    rotate_degree=30,
    crop_size=(16,128,128)
):
    ops = {}
    ops['dflip'] = prob_list_gen(RandomDepthicalFlip3D, probs=[0, 0.5, 1], same_on_batch=False)
    ops['hflip'] = prob_list_gen(RandomHorizontalFlip3D, probs=[0, 0.5, 1], same_on_batch=False)
    ops['vflip'] = prob_list_gen(RandomVerticalFlip3D, probs=[0, 0.5, 1], same_on_batch=False)
    # ops['equal'] = prob_list_gen(RandomEqualize3D, probs=[0, 0.5, 1], same_on_batch=False)

    # affine
    ops['affine'] = [nn.Identity()]
    if isinstance(affine_degree, (float, int)):
        # rotation degree
        affine_degree = [affine_degree]
    if isinstance(affine_shears, (float, int)):
        affine_shears = [affine_shears]
    if isinstance(affine_scale[0], (float, int)):
        # scale, similar to zoom in/out
        affine_scale = [affine_scale]
    for ad_ in affine_degree:
        for ash_ in affine_shears:
            for asc_ in affine_scale:
                affine = prob_list_gen(RandomAffine3D, probs=[0.5, 1], same_on_batch=False, degrees=ad_, scale=asc_, shears=ash_) 
                ops['affine'] += affine

    # random crop
    ops['rcrop'] = []
    if isinstance(crop_size, (float, int)):
        # e.g., crop_size = 32
        crop_size = [(crop_size,)*3]
        rcrop = prob_list_gen(RandomCrop3D, same_on_batch=False, size=crop_size)
    elif isinstance(crop_size[0], (float, int)):
        # e.g., crop_size = (16,64,64)
        crop_size = [crop_size]
    for size in crop_size:
        rcrop = [RandomCrop3D(same_on_batch=False, size=size, p=1)]
        ops['rcrop'] += rcrop

    resize_crop = [nn.Identity()]
    for size in crop_size[:1]:
        size = size[1:]
        for scale in [(0.8, 1), (1, 1)]:
            for ratio in [(1, 1), (3/4, 4/3)]:
                resize_crop += prob_list_gen(RandomResizedCrop3d, probs=[0.5, 1], size=size, scale=scale, ratio=ratio)
    ops['resize_crop'] = resize_crop

    boxblur = [nn.Identity()]
    for ks in [(3,3), (5,5)]:
        boxblur += prob_list_gen(RandomBoxBlur3d, probs=[0.5, 1], kernel_size=ks)
    ops['boxnlur'] = boxblur

    invert = [nn.Identity()]
    for val in [0.25, 0.5, 0.75, 1]:
        invert += prob_list_gen(RandomInvert3d, probs=[0.5, 1], max_val=val)
    ops['invert'] = invert

    gauNoise = [nn.Identity()]
    gauNoise += prob_list_gen(RandomGaussianNoise3d, probs=[0, 0.5, 1])
    ops['gauNoise'] = gauNoise

    erase = [nn.Identity()]
    for scale in [(0.02, 0.1), (0.1, 0.33)]:
        for ratio in [(0.3, 3.3)]:
            erase += prob_list_gen(RandomErasing3d, probs=[0.5, 1], scale=scale, ratio=ratio)
    ops['erase'] = erase

    return ops


class DataAugmentation(BaseNASNetwork):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(
        self,
        rotate_degree=30, crop_size=[(32,128,128), (16,128,128)],
        affine_degree=0, affine_scale=(1.1, 1.5), affine_shears=20,
        mean=0.5, std=0.5,
        mask=None
    ):
        super().__init__(mask)
        self.ops = DAOperation3D(affine_degree, affine_scale, affine_shears, rotate_degree, crop_size)
        transforms = []
        for key, value in self.ops.items():
            transforms.append(OperationSpace(candidates=value, key=key, mask=self.mask, reduction='mean'))
        self.transforms = nn.Sequential(*transforms)
        self.mean = mean
        self.std = std

    def sub_forward(self, x: torch.Tensor, aug=True):
        if aug:
            for idx, trans in enumerate(self.transforms):
                x = trans(x) # BxCXDxHxW
        # normalize
        # Todo: compare with no normalization
        # x = (x-self.mean)/self.std
        return x

    def forward(self, x, aug=True):
        if self.mask is not None:
            with torch.no_grad():
                return self.sub_forward(x, aug)
        else:
            # search mode
            return self.sub_forward(x, aug)

    @property
    def arch(self):
        _arch = []
        for op in self.transforms:
            mask = op.mask
            if 'bool' in str(mask.dtype):
                index = mask.int().argmax()
            else:
                index = mask.float().argmax()
            _arch.append(f"{op.candidates[index]}")
        _arch = '\n'.join(_arch)
        return _arch

if __name__ == '__main__':
    from hyperbox.mutator import DartsMutator, OnehotMutator, RandomMutator
    x = torch.rand(2,1,6,300,300)
    op = DataAugmentation(
        crop_size=[(2,128,128), (3,256,256), (6,200,200)],
        affine_degree=[10, 30],
        affine_scale=[(0.8,1.),(1.1,1.8)],
        affine_shears=[10]
    )
    # m = RandomMutator(op)
    # m = DartsMutator(op)
    m = OnehotMutator(op)
    for i in range(10):
        m.reset()
        print(op.arch)
        y = op(x)
        print(y.shape)
        mask = m.export()
        print(mask)

