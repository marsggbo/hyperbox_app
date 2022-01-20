import os
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import kornia
from einops import rearrange, reduce, repeat
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import *
'''
All augmentation ops are built based on kornia==0.6.2
'''

__all__ = [
    'Base2dTo3d',
    'RandomInvert3d',
    'RandomGaussianNoise3d',
    'RandomBoxBlur3d',
    'RandomErasing3d',
    'RandomSharpness3d',
    'RandomResizedCrop3d'
    'BrightContrast3d'
]


class Base2dTo3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bs = x.shape[0]
        assert len(x.shape) == 5, f"len of x.shape should be 5, i.e., B,C,D,H,W"
        x = rearrange(x, 'b c d h w -> (b d) c h w', b=bs)
        x = self.aug(x)
        x = rearrange(x, '(b d) c h w -> b c d h w', b=bs)
        return x


class BrightContrast3d(nn.Module):
    def __init__(self, brightness=0.4, contrast=0.4, p=0.5):
        super(BrightContrast3d, self).__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.p = p
        self.bright_op = kornia.enhance.adjust_brightness
        self.contr_op = kornia.enhance.adjust_contrast

    def forward(self, x):
        prob = torch.rand(1).item()
        if prob < self.p:
            x = self.bright_op(x, self.brightness)
            x = self.contr_op(x, self.contrast)
        return x

    def __repr__(self):
        return f"BrightContrast3d(brightness={self.brightness}, contrast={self.contrast}, p={self.p})"


class RandomResizedCrop3d(Base2dTo3d):
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.8, 1.0),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (3.0 / 4.0, 4.0 / 3.0),
        resample: Union[str, int] = 'bilinear',
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = "slice",
    ):
        super(RandomResizedCrop3d, self).__init__()
        self.aug = RandomResizedCrop(
            size, scale, ratio, resample, return_transform,
            same_on_batch, align_corners, p, keepdim, cropping_mode
        )


class RandomInvert3d(Base2dTo3d):
    def __init__(
        self,    
        max_val: Union[float, torch.Tensor] = torch.tensor(1.0),
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5
    ):
        '''
        Args:
            max_val: The expected maximum value in the input tensor. The shape has to
            according to the input tensor shape, or at least has to work with broadcasting.
            return_transform: if ``True`` return the matrix describing the transformation applied to each
                input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
            same_on_batch: apply the same transformation across the batch.
            p: probability of applying the transformation.
            keepdim: whether to keep the output shape the same as input (True) or broadcast it
                    to the batch form (False).
        '''
        super(RandomInvert3d, self).__init__()
        self.aug = RandomInvert(max_val, return_transform, same_on_batch, p)


class RandomGaussianNoise3d(Base2dTo3d):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5
    ) -> None:
        super(RandomGaussianNoise3d, self).__init__()
        self.aug = RandomGaussianNoise(mean, std, return_transform, same_on_batch, p)


class RandomBoxBlur3d(Base2dTo3d):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        border_type: str = "reflect",
        normalized: bool = True,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
    ):
        super(RandomBoxBlur3d, self).__init__()
        self.aug = RandomBoxBlur(
        kernel_size, border_type, normalized, return_transform, same_on_batch, p)


class RandomErasing3d(Base2dTo3d):
    def __init__(
        self,
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.02, 0.33),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (0.3, 3.3),
        value: float = 0.0,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ):
        '''
        Args:
            p: probability that the random erasing operation will be performed.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.
            same_on_batch: apply the same transformation across the batch.
            keepdim: whether to keep the output shape the same as input (True) or broadcast it
                            to the batch form (False).
        '''
        super(RandomErasing3d, self).__init__()
        self.aug = RandomErasing(
            scale, ratio, value, return_transform, same_on_batch, p, keepdim)


class RandomSharpness3d(Base2dTo3d):
    def __init__(
        self,
        sharpness: Union[torch.Tensor, float, Tuple[float, float], torch.Tensor] = 0.5,
        same_on_batch: bool = False,
        return_transform: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ):
        super(RandomSharpness3d, self).__init__()
        self.aug = RandomSharpness(
            sharpness, same_on_batch, return_transform, p,  keepdim)


if __name__ == '__main__':
    x = torch.rand(2,3,8,8,8)
    op = RandomBoxBlur3d()
    y = op(x)
    print(y.shape)
    op = RandomInvert3d()
    y = op(x)
    print(y.shape)
    op = RandomGaussianNoise3d()
    y = op(x)
    print(y.shape)
    op = RandomBoxBlur3d()
    y = op(x)
    print(y.shape)
    op = RandomErasing3d()
    y = op(x)
    print(y.shape)
    op = RandomSharpness3d()
    y = op(x)
    print(y.shape)
    print('done')