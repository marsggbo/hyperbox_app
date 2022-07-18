import json
import os
from glob import glob

import torch
import numpy as np
from hydra.utils import instantiate
from hyperbox.engine.base_engine import BaseEngine
from hyperbox.utils.utils import load_json
from omegaconf import DictConfig
from scipy.stats import kendalltau
from hyperbox.utils.logger import get_logger
from hyperbox_app.distributed.utils.twonn import twonn1, twonn2

log = get_logger(__name__)


def plot_ids_by_layers(model_ids, figsize=(8,8), topk=None):
    '''
    Args:
        model_ids: dict. {
            0: {'acc': 0.9288, 'IDs': [13,14,18,35,45,32,16,12,5]},
            1: {'acc': 0.9365, 'IDs': [...]},
            ...
        }
    '''
    import random
    import itertools
    from matplotlib import pyplot as plt
    marker = itertools.cycle(('+', '<',  'd', 'h', 'H','1', '.', '2', 'D', 'o', '*', 'v', '>')) 
    fig = plt.figure(num=1,figsize=figsize)
    ax = fig.add_subplot(111)
    for key, value in model_ids.items():
        if topk is not None and key > topk:
            break
        label, IDs = value['label'], value['IDs']
        label = f"{key}_{label}"
        x_axis = np.array(list(range(len(IDs))))/(len(IDs)-1)
        y_axis = IDs
        color = (random.random(), random.random(), random.random())
        # print(x_axis, y_axis)
        ax.plot(x_axis, y_axis, color=color, marker=next(marker), label=label)
    ax.legend()
    plt.savefig('model_ids.pdf')
    plt.show()


class IDAnalysis(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        search_space_path: str=None,
        twonn_alg: str='2',
        num_subnets: int=20,
        *args, **kwargs
    ):
        super().__init__(trainer, model, datamodule, cfg)
        self.search_space_path = search_space_path
        if self.search_space_path is None:
            self.masks = None
        elif 'json' in search_space_path:
            with open(search_space_path, 'r') as f:
                self.masks = json.load(f)
        else:
            self.masks = glob(f"{search_space_path}/*.json")
        self.twonn_alg = str(twonn_alg)
        self.num_subnets = num_subnets
        if self.twonn_alg == '1':
            self.twonn = twonn1
        elif self.twonn_alg == '2':
            self.twonn = twonn2

    def parse_arch(self, pl_module):
        if hasattr(pl_module, 'arch_encoding'):
            arch = pl_module.arch_encoding
        elif hasattr(pl_module.network, 'arch_encoding'):
            arch = pl_module.network.arch_encoding
        else:
            raise NotImplementedError
        return arch

    def convert_list2tensor(self, src):
        for key, val in src.items():
            if isinstance(val, list):
                src[key] = torch.tensor(val)
        return src

    def run(self):
        # self.trainer.limit_val_batches = 2
        self.datamodule.setup()
        if 'CIFAR10DataModule' in self.datamodule.__class__.__name__:
            dataset = 'cifar10'
        elif 'CIFAR100DataModule' in self.datamodule.__class__.__name__:
            dataset = 'cifar100'

        valid_loader = self.datamodule.val_dataloader()
        x, y = next(iter(valid_loader))
        batch = x.shape[0]

        self.model_ids = {}
        num_subnets = self.num_subnets
        if self.masks is not None:
            keys = list(self.masks.keys())
            if self.num_subnets == '-1':
                num_subnets = len(self.masks)
            else:
                print(keys[:num_subnets])
        for i in range(num_subnets):
            self.model_ids[i] = {'label': '', 'IDs': []}
            if self.masks is not None:
                mask = self.convert_list2tensor(self.masks[keys[i]])
                self.model.mutator.sample_by_mask(mask)
            else:
                self.model.mutator.reset()
            y = self.model(x)
            features = self.model.network.features
            acc = self.model.network.query_by_key()
            IDs = []
            for idx, feat in enumerate(features):
                _id = self.twonn(feat.view(batch, -1).cpu().detach().numpy())
                IDs.append(_id)
            self.model_ids[i]['IDs'] = IDs
            net_name = self.model.network.__class__.__name__
            alg_name = self.twonn_alg
            self.model_ids[i]['label'] = f'nb201_ID{alg_name}_subnet{i}_acc{acc:.4f}'
            print(f'{net_name}_{alg_name}_subnet{i}: {IDs}')
        plot_ids_by_layers(self.model_ids)

