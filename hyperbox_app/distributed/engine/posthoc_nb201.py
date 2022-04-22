import json
import os
from glob import glob

import torch
from hydra.utils import instantiate
from hyperbox.engine.base_engine import BaseEngine
from hyperbox.utils.utils import load_json
from omegaconf import DictConfig
from scipy.stats import kendalltau
from hyperbox.utils.logger import get_logger

log = get_logger(__name__)


class PosthocNB201Engine(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        search_space_path: str,
        num_subnets: int=100,
        metric_key: str='val/acc_epoch',
        query_key: str='test_acc'
    ):
        super().__init__(trainer, model, datamodule, cfg)
        self.metric_key = metric_key
        self.query_key = query_key
        self.num_subnets = int(num_subnets)
        self.search_space_path = search_space_path
        if 'json' in search_space_path:
            with open(search_space_path, 'r') as f:
                self.masks = json.load(f)
        else:
            self.masks = glob(f"{search_space_path}/*.json")
        self.idx = 0
        self.performance_history = {}

    def reset_idx(self):
        self.idx = 0

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
        self.performance_history = {}

        while self.idx < self.num_subnets:
            if isinstance(self.masks, list):
                mask = load_json(self.masks[self.idx])
            else:
                key = list(self.masks.keys())[self.idx]
                mask = self.convert_list2tensor(self.masks[key])
            self.model.mutator.sample_by_mask(mask)
            self.model.network.sync_mask_for_all_cells(mask)
            arch = self.model.network.arch_encoding
            metrics = self.trainer.validate(model=self.model, datamodule=self.datamodule, verbose=False)
            metrics = metrics[0][self.metric_key]
            if arch not in self.performance_history:
                self.performance_history[arch] = {'proxy': 0, 'real': 0}
            self.performance_history[arch]['proxy'] = metrics
            real = self.model.network.query_by_key(key=self.query_key, dataset=dataset)
            self.performance_history[arch]['real'] = real
            log.info(f"{self.idx}-{arch}: {metrics} {real}")
            self.idx += 1
        proxies = []
        reals = []
        for arch in self.performance_history:
            proxies.append(self.performance_history[arch]['proxy'])
            reals.append(self.performance_history[arch]['real'])
        tau, p = kendalltau(proxies, reals)
        log.info(f"#valid_batches={self.trainer.limit_val_batches}")
        log.info(f"Kendall's Tau: {tau}, p-value: {p}")
        with open(
            '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/engine/nb201_results.csv',
            'a') as f:
            gpus = self.cfg.trainer.gpus
            mode = 'hete' if self.cfg.model.is_net_parallel else 'homo'
            bs = self.trainer.limit_val_batches
            num_nets = self.num_subnets
            pw = self.cfg.pretrained_weight
            # #GPUs, Hete/Homo, #EvalBatch, #nets, dataset, PretrainedWeights, Tau, P-value
            f.write(f"\n{gpus}, {mode}, {bs}, {num_nets}, {dataset}, {pw}, {tau}, {p}")
        return {'tau': tau, 'p': p}
