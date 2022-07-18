import json
import os
from glob import glob
from types import MethodType

import numpy as np
import torch
from hydra.utils import instantiate
from hyperbox.engine.base_engine import BaseEngine
from hyperbox.mutator import EAMutator, EvolutionMutator
from hyperbox.mutator.utils import NonDominatedSorting
from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import load_json
from omegaconf import DictConfig
from scipy.stats import kendalltau

log = get_logger(__name__)


def query_api(self, arch):
    '''
    self -> mutator
    self.model -> pytorch model
    self.pl_model -> pytorch-lightning module
    '''
    real = self.model.query_by_key(self.query_file_path, self.query_key)
    proxy = self.__dict__['trainer'].validate(
        model=self.__dict__['pl_model'], datamodule=self.__dict__['datamodule'], verbose=False)[0][self.metric_key]
    encoding = self.arch2encoding(arch)
    info = self.vis_dict[encoding]
    info['real_perf'] = real
    info['proxy_perf'] = proxy
    return proxy


class EASearchNBMBNet(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        query_file_path: str,
        metric_key: str='val/acc_epoch',
        query_key: str='mean_acc'
    ):
        super().__init__(trainer, model, datamodule, cfg)
        assert isinstance(model.mutator, (EAMutator, EvolutionMutator))
        self.mutator = model.mutator
        self.query_file_path = query_file_path
        self.metric_key = metric_key
        self.query_key = query_key
        self.performance_history = {}

    def convert_list2tensor(self, src):
        for key, val in src.items():
            if isinstance(val, list):
                src[key] = torch.tensor(val)
        return src

    def run(self):
        # self.trainer.limit_val_batches = 50
        self.datamodule.setup()
        if 'CIFAR10DataModule' in self.datamodule.__class__.__name__:
            dataset = 'cifar10'
        elif 'CIFAR100DataModule' in self.datamodule.__class__.__name__:
            dataset = 'cifar100'
        self.performance_history = {}

        mutator = self.mutator
        self.mutator.__dict__['trainer'] = self.trainer
        self.mutator.__dict__['pl_model'] = self.model
        self.mutator.__dict__['datamodule'] = self.datamodule
        self.mutator.query_api = MethodType(query_api, self.mutator)
        self.mutator.metric_key = self.metric_key
        self.mutator.query_key = self.query_key
        self.mutator.query_file_path = self.query_file_path
        try:
            self.mutator.search()

            reals = [info['real_perf'] for encoding, info in self.mutator.vis_dict.items() if 'real_perf' in info]
            proxies = [info['proxy_perf'] for encoding, info in self.mutator.vis_dict.items() if 'proxy_perf' in info]
            tau_visited, p_visited = kendalltau(reals, proxies)
            self.mutator.plot_real_proxy_metrics(
                real_metrics=reals,
                proxy_metrics=proxies,
                figname='evolution_visited_real_proxy_metrics.png')

            reals = [info['real_perf'] for info in self.mutator.keep_top_k[self.mutator.topk] if 'real_perf' in info]
            proxies = [info['proxy_perf'] for info in self.mutator.keep_top_k[self.mutator.topk] if 'proxy_perf' in info]
            tau_topk, p_topk = kendalltau(reals, proxies)
            self.mutator.plot_real_proxy_metrics(
                real_metrics=reals,
                proxy_metrics=proxies,
                figname=f'evolution_top{self.mutator.topk}_real_proxy_metrics.png')


            results = {
                'tau_visited': tau_visited,
                'p_visited': p_visited,
                'tau_topk': tau_topk,
                'p_topk': p_topk,
            } 
            log.info(results)

            with open(
                '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/engine/nbmbnet_ea_search.csv',
                'a') as f:
                #K(subnets), #B(sample_interval), dataset, PretrainedWeights, seed, #valBatch, Tau_visited, P_visited, Tau_topk, P_topk
                valBatch = self.trainer.limit_val_batches
                pw = self.cfg.pretrained_weight
                K = pw.split('net')[0].split('_')[-1]
                B = pw.split('batch')[0].split('_')[-1]
                seed = self.cfg.seed
                text = f"{K},{B},{tau_visited},{p_visited},{tau_topk},{p_topk},{dataset},{seed},{valBatch},{pw}\n"
                # text = f"{K},{B},{dataset},{pw},{seed},{valBatch},{tau_visited},{p_visited},{tau_topk},{p_topk}\n"
                f.write(text)
                log.info(text)
            return results
        except Exception as e:
            self.mutator.save_checkpoint()
            return {'error_info': str(e)}
