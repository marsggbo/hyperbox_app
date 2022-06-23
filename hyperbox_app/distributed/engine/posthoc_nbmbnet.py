import json
import os
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


class PosthocNBMBNetEngine(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        search_space_path: str,
        query_file_path: str,
        metric_key: str='val/acc_epoch',
        query_key: str='mean_acc',
        logpath: str=None,
    ):
        super().__init__(trainer, model, datamodule, cfg)
        self.metric_key = metric_key
        self.query_key = query_key
        self.query_file_path = query_file_path
        self.search_space_path = search_space_path
        if logpath is not None:
            self.logpath = logpath
        else:
            self.logpath = os.path.join(os.getcwd(), 'evaltau_nbmbnet.csv')
        if 'json' in search_space_path:
            with open(search_space_path, 'r') as f:
                self.masks = json.load(f)
        else:
            self.masks = glob(f"{search_space_path}/*.json")
        self.idx = 0
        self.performance_history = {}

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

        num = len(self.masks)
        for i in range(100):
            if isinstance(self.masks, list):
                mask = load_json(self.masks[self.idx])
            else:
                key = list(self.masks.keys())[self.idx]
                # mask = self.convert_list2tensor(self.masks[key])
                mask = self.key2mask(key, self.masks[key])
            self.model.mutator.sample_by_mask(mask)
            arch = self.model.network.arch
            metrics = self.trainer.validate(model=self.model, datamodule=self.datamodule, verbose=False)
            metrics = metrics[0][self.metric_key]
            if arch not in self.performance_history:
                self.performance_history[arch] = {'proxy': 0, 'real': 0}
            self.performance_history[arch]['proxy'] = metrics
            real = self.model.network.query_by_key(self.query_file_path, key=self.query_key)
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
            pw = self.cfg.pretrained_weight
            num_nets = pw.split('net')[0].split('_')[-1]
            interval = pw.split('batch')[0].split('_')[-1]
            gpus = pw.split('gpunum')[1].split('_')[0]
            bs = self.trainer.limit_val_batches
            # #GPUs, Hete/Homo, #EvalBatch, #nets, dataset, PretrainedWeights, Tau, P-value
            f.write(f"\n{num_nets}, {interval}, {tau}, {p}, {pw}, {gpus}, {bs}")
        return {'tau': tau, 'p': p}
