import json
import os
import pickle
from glob import glob

import numpy as np
import torch
from hydra.utils import instantiate
from hyperbox.engine.base_engine import BaseEngine
from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import load_json
from omegaconf import DictConfig
from scipy.stats import kendalltau

log = get_logger(__name__)


class FewshotEval(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        supernet_mask_path_pattern: str,
        search_space_path: str,
        ckpts_path_pattern: str,
        query_file_path: str=None,
        metric_key: str='val/acc_epoch',
        query_key: str='mean_acc',
        logpath: str=None,
    ):
        super().__init__(trainer, model, datamodule, cfg)
        self.metric_key = metric_key
        self.query_key = query_key
        self.supernet_mask_paths = glob(supernet_mask_path_pattern)
        self.supernet_masks = []
        for path in self.supernet_mask_paths:
            self.supernet_masks.append(load_json(path))

        self.query_file_path = query_file_path
        self.ckpts_path_pattern = ckpts_path_pattern
        self.ckpts_path = glob(ckpts_path_pattern)
        self.search_space_path = search_space_path
        if logpath is not None:
            self.logpath = logpath
        else:
            self.logpath = os.path.join(os.getcwd(), 'evaltau.csv')
        if 'json' in search_space_path:
            with open(search_space_path, 'r') as f:
                self.masks = json.load(f)
        else:
            self.masks = glob(f"{search_space_path}/*.json")
        self.mutator = self.model.mutator
        self.search_space_to_eval = self.partition_search_space(masks=self.masks)
        self.idx = 0
        self.performance_history = {}

    def is_subnet_in_supernet(self, subnet_mask, supernet_mask):
        enc_supernet = torch.vstack([v.bool() for v in supernet_mask.values()])
        enc_subnet = torch.vstack([v.bool() for v in subnet_mask.values()])
        return (torch.bitwise_or(enc_supernet, enc_subnet)==enc_supernet).all()

    def partition_search_space(self, masks):
        search_space = {}
        for key, val in masks.items():
            mask = self.convert_list2tensor(val['mask'])
            for space_idx, supernet_mask in enumerate(self.supernet_masks):
                if self.is_subnet_in_supernet(mask, supernet_mask):
                    if space_idx not in search_space:
                        search_space[space_idx] = [mask]
                    else:
                        search_space[space_idx].append(mask)
        return search_space

    def convert_list2tensor(self, src):
        for key, val in src.items():
            if isinstance(val, list):
                src[key] = torch.tensor(val)
        return src

    def key2mask(self, key, value):
        mask = {}
        idx = 0
        if value.get('mask', None) is not None:
            return self.convert_list2tensor(value['mask'] )
        for m in self.model.mutator.mutables:
            k = int(key[idx])
            mask[m.key] = torch.nn.functional.one_hot(torch.tensor(k), m.length).bool()
            idx += 1
        return mask

    def run(self):
        # self.trainer.limit_val_batches = 50
        self.datamodule.setup()
        self.performance_history = {}

        for space_idx in self.search_space_to_eval:
            path = self.ckpts_path[space_idx]
            # path = '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_12net_1batch/2022-05-12_02-38-38/checkpoints/last.ckpt'
            self.model.network.load_from_ckpt(path)
            num = len(self.search_space_to_eval[space_idx])
            for i in range(num):
                mask = self.search_space_to_eval[space_idx][i]
                self.model.mutator.sample_by_mask(mask)
                arch = f"{self.model.network.arch}"
                metrics = self.trainer.validate(model=self.model, datamodule=self.datamodule, verbose=False)
                metrics = metrics[0][self.metric_key]
                if arch not in self.performance_history:
                    self.performance_history[arch] = {'proxy': 0, 'real': 0}
                self.performance_history[arch]['proxy'] = metrics
                real = self.model.network.query_by_key()
                self.performance_history[arch]['real'] = real
                log.info(f"{self.idx}-{arch}: {metrics} {real}")
                self.idx += 1
        proxies = []
        reals = []
        for arch in self.performance_history:
            proxies.append(self.performance_history[arch]['proxy'])
            reals.append(self.performance_history[arch]['real'])
        indices = np.argsort(proxies)
        proxies = np.array(proxies)[indices]
        reals = np.array(reals)[indices]
        tau, p = kendalltau(proxies, reals)
        log.info(f"#valid_batches={self.trainer.limit_val_batches}")
        log.info(f"Kendall's Tau: {tau}, p-value: {p}")
        with open(self.logpath, 'a') as f:
            f.write(f"{tau}, {p}, {len(self.search_space_to_eval)}, {self.trainer.limit_val_batches}\n")
            # pw = self.cfg.pretrained_weight
            # num_nets = pw.split('net')[0].split('_')[-1]
            # interval = pw.split('batch')[0].split('_')[-1]
            # gpus = pw.split('gpunum')[1].split('_')[0]
            # bs = self.trainer.limit_val_batches
            # # #GPUs, Hete/Homo, #EvalBatch, #nets, dataset, PretrainedWeights, Tau, P-value
            # f.write(f"\n{num_nets}, {interval}, {tau}, {p}, {pw}, {gpus}, {bs}")
        return {'tau': tau, 'p': p}
