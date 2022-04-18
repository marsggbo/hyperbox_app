from glob import glob
import json
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.stats import kendalltau

from hyperbox.engine.base_engine import BaseEngine
from hyperbox.utils.utils import load_json


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
        query_key='mean_acc',
    ):
        super().__init__(trainer, model, datamodule, cfg)
        self.metric_key = metric_key
        self.query_key = query_key
        self.query_file_path = query_file_path
        self.search_space_path = search_space_path
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

    def run(self):
        self.trainer.limit_val_batches = 2
        self.datamodule.setup()
        self.performance_history = {}

        for i in range(10):
            self.model.mutator.sample_by_mask(load_json(self.masks[self.idx]))
            self.idx += 1
            arch = self.model.network.arch
            metrics = self.trainer.validate(model=self.model, datamodule=self.datamodule, verbose=False)
            metrics = metrics[0][self.metric_key]
            if arch not in self.performance_history:
                self.performance_history[arch] = {'proxy': 0, 'real': 0}
            self.performance_history[arch]['proxy'] = metrics
            real = self.model.network.query_by_key(self.query_file_path, key=self.query_key)
            self.performance_history[arch]['real'] = real
            print(f"{arch}: {metrics} {real}")
        proxies = []
        reals = []
        for arch in self.performance_history:
            proxies.append(self.performance_history[arch]['proxy'])
            reals.append(self.performance_history[arch]['real'])
        tau, p = kendalltau(proxies, reals)
        print(f"Kendall's Tau: {tau}, p-value: {p}")
        return {'tau': tau, 'p': p}
