import random

import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

def pil_loader(path):
    img = Image.open(path)
    bands = img.getbands()
    if len(bands) >= 3:
        img = img.convert('RGB')
    return img

# def pil_loader(path):
#     return cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     if 'tif' in path.lower():
#         img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     else:
#         img = Image.open(path)
#         bands = img.getbands()
#         if len(bands) >= 3:
#             img = img.convert('RGB')
#         else:
#             img = img.convert('L')
#     return img


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img, depth, height, width):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = depth
    desired_width = width
    desired_height = height
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

class Resampler(object):
    def __init__(self):
        pass

    @classmethod
    def resample(self, slices, threshold):
        '''
        Args:
            slices: the list of slices that requires upsampling.
            threshold: the expected number of slices
        '''
        if threshold == len(slices):
            return slices
        elif threshold > len(slices):
            return self.upsample(slices, threshold)
        else:
            return self.undersample(slices, threshold)

    @staticmethod
    def upsample(slices, threshold=64):
        raise NotImplementedError

    @staticmethod
    def undersample(slices, threshold=64):
        raise NotImplementedError


class RandomResampler(Resampler):
    @staticmethod
    def upsample(slices, threshold=64):
        original_num = len(slices)
        d = threshold - original_num
        tmp = []
        idxs = []
        for _ in range(d):
            idx = random.randint(0, original_num-1)
            idxs.append(idx)
        for idx, value in enumerate(slices):
            tmp.append(value)
            while idx in idxs:
                idxs.remove(idx)
                tmp.append(value)
        return tmp

    @staticmethod
    def undersample(slices, threshold=64):
        original_num = len(slices)
        d = original_num - threshold
        tmp = slices.copy()
        for _ in range(d):
            idx = random.randint(0, len(tmp)-1)
            tmp.pop(idx)
        return tmp


class SymmetricalResampler(Resampler):
    '''
    Examples:
        ```
        a = list(range(7))
        re = SymmetricalResampler()
        re.resample(a,15) # [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6]
        re.resample(a,10) # [0, 1, 1, 2, 3, 3, 4, 5, 5, 6]
        re.resample(a,3) # [1, 3, 5]

        ```
    '''
    @staticmethod
    def upsample(slices, threshold=64):
        tmp = []
        original_num = len(slices)
        add_num = threshold - original_num
        add_idxs = list(range(original_num))
        if add_num % original_num == 0:
            repetitions = add_num // original_num
        else:
            repetitions = add_num // original_num + 1

        if repetitions > 1:
            for _ in range(repetitions-1): add_idxs.extend(list(range(original_num)))

        remain_num = threshold - len(add_idxs)
        if remain_num == 0:
            return tmp
        else:
            interval = original_num // remain_num
            idx = original_num // 2 if original_num%2==1 else original_num//2 -1
            remain_list = [idx]
            count = 1
            flag = 'right'
            while len(remain_list) < remain_num:
                if flag == 'right':
                    idx = idx + count * interval
                    flag = 'left'
                elif flag == 'left':
                    idx = idx - count * interval
                    flag = 'right'
                count += 1
                remain_list.append(idx)
            add_idxs.extend(remain_list)
            for idx in sorted(add_idxs):
                tmp.append(slices[idx])
            return tmp

    @staticmethod
    def undersample(slices, threshold=64):
        tmp = []
        original_num = len(slices)
        add_idxs = []
        remain_num = threshold
        interval = original_num // remain_num
        idx = original_num // 2 if original_num%2==1 else original_num//2 -1
        remain_list = [idx]
        count = 1
        flag = 'right'
        while len(remain_list) < remain_num:
            if flag == 'right':
                idx = idx + count * interval
                flag = 'left'
            elif flag == 'left':
                idx = idx - count * interval
                flag = 'right'
            count += 1
            remain_list.append(idx)
        add_idxs.extend(remain_list)
        for idx in sorted(add_idxs):
            tmp.append(slices[idx])
        return tmp
