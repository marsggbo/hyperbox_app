import json
import os
import random
from typing import List, Optional, Union

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as TF
from hyperbox.utils.utils import hparams_wrapper
from hyperbox_app.medmnist.datamodules.utils import (RandomResampler,
                                                     SymmetricalResampler,
                                                     normalize, pil_loader,
                                                     resize_volume)
from kornia import image_to_tensor, tensor_to_image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler

from time import time

__all__ = [
    'CTDataset',
    'CTDatamodule'
]


@hparams_wrapper
class CTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        data_list: str,
        is_train: bool,
        is_color: bool=False,
        is_3d: bool=True,
        img_size: Union[List, int]=[512,512],
        center_size: Union[List, int]=[360,360],
        slice_num: int=64,
        loader=pil_loader,
        transforms=None,
        label_transforms=None,
        use_weighted_sampler: bool = False,
        use_balanced_batch_sampler: bool = False,
        *args, **kwargs
    ):
        '''
        Args:
            root_dir: root dir of dataset, e.g., ~/../../datasets/CCCCI_cleaned/dataset_cleaned/
            data_list: the training of testing data list or json file. e.g., ct_train.json
            is_train: determine to load which type of dataset
            slice_num: the number of slices in a scan
        '''
        with open(self.data_list, 'r') as f:
            self.data = json.load(f)
        self.cls_to_label = {
            # png slices
            'CP': 0, 'NCP': 1, 'Normal': 2,
            # nii
            'CT-0': 0, 'CT-1': 1, 'CT-2': 1, 'CT-3': 1, 'CT-4': 1,
            # covid_ctset
            'normal': 0, 'covid': 1
        }
        self.samples = self.convert_json_to_list(self.data)
        self.count = 1000

    def convert_json_to_list(self, data):
        samples = {} # {0: {'scans': [], 'labels': 0}}
        labels = []
        idx = 0
        for cls_ in data:
            for pid in data[cls_]:
                for scan_id in data[cls_][pid]:
                    slices = data[cls_][pid][scan_id]
                    label = self.cls_to_label[cls_]
                    if slices[0].endswith('.nii') or slices[0].endswith('.gz'):
                        scan_path = os.path.join(self.root_dir,cls_,slices[0])
                    else:
                        scan_path = os.path.join(self.root_dir,cls_,pid,scan_id)
                    if os.path.exists(scan_path) and len(slices)>0:
                            samples[idx] = {'slices':slices, 'label': label, 'path': scan_path}
                            labels.append(label)
                            idx += 1
        self.labels = torch.tensor(labels).view(-1)
        return samples

    def preprocessing(self, img):
        # resize = int(self.img_size[0]*5/4)
        resize = int(self.img_size[0])
        if isinstance(img, np.ndarray):
            img = cv2.resize(img, (resize, resize))
            center = int(resize/2)
            half_size = self.center_size[0] // 2
            h_left, h_right = center - half_size,  center + half_size
            w_left, w_right = center - half_size,  center + half_size
            img = img[h_left:h_right, w_left:w_right]
            img = TF.ToTensor()(img)
        else:
            transform = TF.Compose([
                TF.Resize(self.img_size),
                TF.CenterCrop(self.center_size),
                TF.ToTensor()
            ])
            img = transform(img)
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = torch.tensor(sample['label']).long()
        # stack & sample slice
        if sample['slices'][0].endswith('.nii') or sample['slices'][0].endswith('.nii.gz'):
            slice_tensor = self.get_nifti(sample)
        else:
            slice_tensor = self.get_png(sample)

        # if not 3d, then remove channel dimension
        if not self.is_3d: slice_tensor = slice_tensor[0, :, :, :]
        return slice_tensor, label
        # return slice_tensor, label, sample['path']

    def get_nifti(self, sample):
        start = time()
        path = sample['path']
        slice_tensor = []
        slice_path = path
        img = nib.load(slice_path) 
        end_load = time()
        if self.count <= 100:
            print(f'loading data costs {end_load-start} s')
        img_fdata = img.get_fdata()
        (x,y,z) = img.shape
        img_fdata = normalize(img_fdata)
        end_norm = time()
        if self.count <= 100:
            print(f'norm data costs {end_norm-end_load} s')

        # # sample method 1: zoom
        # depth, height, width = self.slice_num, self.img_size[0], self.img_size[1]
        # slice_tensor = resize_volume(img_fdata, depth, height, width)
        # end_zoom = time()
        # if self.count <= 100:
        #     print(f'zoom data costs {end_zoom-end_norm} s')
        # slice_tensor = torch.from_numpy(slice_tensor).float().unsqueeze(dim=0)
        # slice_tensor = slice_tensor.permute(0, 3, 1, 2)
        # end = time()
        # if self.count <= 100:
        #     print(f'once loading data costs {end-start} s')
        #     self.count += 1

        # sample method 2: depth sampling
        slice_tensor = torch.FloatTensor(img_fdata)
        slice_tensor = slice_tensor.unsqueeze(dim=0)
        slice_tensor = slice_tensor.permute(0, 3, 1, 2)
        if self.is_train:
            slices = RandomResampler.resample(list(range(z)), self.slice_num)
        else:
            slices = SymmetricalResampler.resample(list(range(z)), self.slice_num)
        slice_tensor = slice_tensor[:, slices, :, :]
        # todo: imbalanced problem
        h, w = self.img_size[0], self.img_size[1]
        size = (h*5//4, w*5//4)
        center_h, center_w = size[0]//2, size[1]//2
        slice_tensor = torch.nn.functional.interpolate(slice_tensor, size) # resize
        slice_tensor = slice_tensor[:, :, center_h-h//2:center_h+h//2, center_w-w//2:center_w+w//2] # centercrop

        return slice_tensor

    def get_png(self, sample):
        path = sample['path']
        if self.is_train:
            slices = RandomResampler.resample(sample['slices'], self.slice_num)
        else:
            slices = SymmetricalResampler.resample(sample['slices'], self.slice_num)

        slice_tensor = []
        for slice_ in slices:
            slice_path = os.path.join(path, slice_)
            img = self.loader(slice_path) # height * width
            img = self.preprocessing(img) # 1 * height * width
            if not self.is_color:
                if len(img.shape)==3:
                    img = torch.unsqueeze(img[0, :, :], dim=0)
                else:
                    img = torch.unsqueeze(img, dim=0)
            slice_tensor.append(img)
        slice_tensor = torch.stack(slice_tensor)
        slice_tensor = slice_tensor.permute(1, 0, 2, 3) # c*d*h*w

        return slice_tensor

    def __len__(self):
        return len(self.samples)

    def dataset_name(self):
        root = self.root_dir.lower()
        if 'cccc' in root:
            name = 'CCCCII'
        elif 'mosmed' in root:
            name = 'MosMed'
        elif 'covid' in root:
            name = 'Iran-COVID'
        return name

    def __repr__(self):
        name = self.dataset_name()
        return f"{super(CTDataset, self).__repr__()} ({name})"


@hparams_wrapper
class CTDatamodule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        data_list_train: str,
        data_list_val: str,
        data_list_test: str,
        is_color: bool=True,
        is_3d: bool=True,
        img_size: Union[List, int]=[512, 512],
        center_size: Union[List, int]=[360, 360],
        batch_size: int=16,
        slice_num: int=32,
        seed: int = 666,
        is_customized: bool = False,
        num_workers: int = 4,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        use_weighted_sampler: bool = False,
        use_balanced_batch_sampler: bool = False,
        class_weights: list = None
    ):
        super().__init__()
        self.is_setup = False
        self.setup()
        self.use_weighted_sampler = use_weighted_sampler
        self.class_weights = class_weights

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return
        self.dataset_train = CTDataset(
            self.root_dir, data_list=self.data_list_train, is_train=True, is_color=self.is_color,
            is_3d=self.is_3d, img_size=self.img_size, center_size=self.center_size, slice_num=self.slice_num)
        if self.is_customized:
            self.dataset_val = self.dataset_train
        else:
            self.dataset_val = CTDataset(
                self.root_dir, data_list=self.data_list_val, is_train=False, is_color=self.is_color,
                is_3d=self.is_3d, img_size=self.img_size, center_size=self.center_size, slice_num=self.slice_num)
        self.dataset_test = CTDataset(
            self.root_dir, data_list=self.data_list_test, is_train=False, is_color=self.is_color,
            is_3d=self.is_3d, img_size=self.img_size, center_size=self.center_size, slice_num=self.slice_num)
        self.datasets = [
            self.dataset_train, self.dataset_val, self.dataset_test
        ]
        self.is_setup = True

    def build_weighted_sampler(self, dataset, class_weights: list=None):
        '''
        class_weights: class_weights list
        '''
        labels = torch.tensor(dataset.labels).view(-1)
        if class_weights is None:
            class_sample_count = torch.tensor(
                [(labels == t).sum() for t in torch.unique(labels, sorted=True)])
            class_weights = 1. / class_sample_count.float()
            print(class_weights)
        samples_weight = torch.tensor([class_weights[t.item()] for t in labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight)*2, replacement=True)
        return sampler

    def _data_loader(self, dataset: Dataset, shuffle: bool = False, sampler=None) -> DataLoader:
        if sampler is not None:
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=sampler
        )

    def train_dataloader(self):
        tr_sampler = None
        if self.use_balanced_batch_sampler:
            from catalyst.data import BalanceClassSampler, BatchBalanceClassSampler
            labels = self.dataset_train.labels
            num_classes = len(set(np.array(labels)))
            num_samples = self.batch_size
            num_batches = max(50,  len(labels) // (num_classes * num_samples))
            sampler = BatchBalanceClassSampler(
                labels.tolist(), num_classes=num_classes, num_samples=num_samples, num_batches=num_batches)
            # sampler = DistributedSamplerWrapper(sampler)
            train_loader = DataLoader(
                dataset=self.dataset_train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                batch_sampler=sampler,
            )
        elif self.use_weighted_sampler:
            weights = self.class_weights
            tr_sampler = self.build_weighted_sampler(self.dataset_train, weights)
            train_loader = self._data_loader(self.dataset_train, True, tr_sampler)
        else:
            train_loader = self._data_loader(self.dataset_train, True, None)
        if self.is_customized:
            val_sampler = None
            if self.use_weighted_sampler:
                weights = self.class_weights
                val_sampler = self.build_weighted_sampler(self.dataset_val, weights)
            train_val_loader = {
                'train': train_loader,
                'val': self._data_loader(self.dataset_val, True, val_sampler)
            }
            return train_val_loader
        return train_loader

    def val_dataloader(self):
        if self.is_customized:
            return self.test_dataloader()
        return self._data_loader(self.dataset_val)

    def test_dataloader(self):
        return self._data_loader(self.dataset_test)

    def change_img_size(self, img_size):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        for dataset in self.datasets:
            dataset.img_size = img_size

    def change_slice_num(self, slice_num):
        for dataset in self.datasets:
            dataset.slice_num = slice_num

    def __repr__(self):
        name = self.dataset_test.dataset_name()
        return f"{super(CTDatamodule, self).__repr__()} ({name})"


if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.INFO)
    # dataset_cccci = CTDataset(
    #     root_dir='/home/datasets/CCCCI_cleaned/dataset_cleaned/',
    #     data_list='/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datasets/ccccii/ct_train.json',
    #     is_train=True
    # )
    # dataset_nii = CTDataset(
    #     root_dir='/home/datasets/MosMedData/COVID19_1110/pngs',
    #     data_list='/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datasets/mosmed/nii_png_train.json',
    #     is_train=True
    # )
    # dataset_ct = CTDataset(
    #     root_dir='/home/datasets/COVID-CTset_visual',
    #     data_list='/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datasets/covid_ctset/train_balance.json',
    #     is_train=True
    # )
    # for dataset in [dataset_cccci, dataset_nii, dataset_ct]:
    #     data = dataset[0]
    #     logging.info(data[0].shape)

    BS = 32
    print('start')
    logging.info('start')
    use_weighted_sampler = True
    # mosmed
    # mean=0.4570799767971039, std=0.1314811408519745
    datamodule1 = CTDatamodule(
        root_dir='/home/datasets/MosMedData/COVID19_1110/pngs',
        is_color=False,
        batch_size=BS,
        num_workers=4,
        use_weighted_sampler=use_weighted_sampler,
        class_weights=[1/178, 1/601],
        data_list_train='/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/datamodules/mosmed/nii_png_train.json',
        data_list_val='/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/datamodules/mosmed/nii_png_test.json',
        data_list_test='/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/datamodules/mosmed/nii_png_test.json',
    )
    # ccccii
    # mean=0.5486013889312744, std=0.37949830293655396
    datamodule2 = CTDatamodule(
        root_dir='/home/datasets/CCCCI_cleaned/dataset_cleaned/',
        is_color=False,
        batch_size=BS,
        num_workers=4,
        use_weighted_sampler=use_weighted_sampler,
        class_weights=[1/1210, 1/1213,1/772],
        data_list_train='/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/datamodules/ccccii/ct_train.json',
        data_list_val='/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/datamodules/ccccii/ct_test.json',
        data_list_test='/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/datamodules/ccccii/ct_test.json',
    )
    # iran
    # mean=0.24609440565109253, std=0.16903874278068542
    datamodule3 = CTDatamodule(
        root_dir='/home/datasets/COVID-CTset_visual',
        is_color=False,
        batch_size=BS,
        num_workers=4,
        use_weighted_sampler=use_weighted_sampler,
        class_weights=[1/236, 1/595],
        data_list_train='/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/datamodules/iran/train.json',
        data_list_val='/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/datamodules/iran/test.json',
        data_list_test='/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/datamodules/iran/test.json',
    )
    logging.info('dataset done')
    from time import time
    start = time()
    for datamodule in [datamodule1, datamodule2, datamodule3]:
        mean = 0.
        std = 0.
        labels = []
        for idx, data in enumerate(datamodule.train_dataloader()):
            # if idx > 2:
            #     break
            img, label = data
            labels.append(label)
            # mean += img.view(img.shape[0], -1).mean()
            # std += img.view(img.shape[0], -1).std()
        # mean /= (idx+1)
        # std /= (idx+1)
        # logging.info(f"mean={mean}, std={std}")
        labels = torch.cat(labels).view(-1)
        num0 = (labels==0).sum()
        num1 = (labels==1).sum()
        num2 = (labels==2).sum()
        print(f"0:{num0} 1:{num1} 2:{num2} ")
    cost = time() - start
    logging.info(f"cost {cost/(idx+1)} sec")
    logging.info('end')
