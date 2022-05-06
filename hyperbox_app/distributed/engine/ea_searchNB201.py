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
    real = self.model.query_by_key(self.query_key)
    proxy = self.__dict__['trainer'].validate(
        model=self.__dict__['pl_model'], datamodule=self.__dict__['datamodule'], verbose=False)[0][self.metric_key]
    encoding = self.arch2encoding(arch)
    info = self.vis_dict[encoding]
    info['real_perf'] = real
    info['proxy_perf'] = proxy
    return proxy


class EASearchNB201(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        sample_iterations: int=1000,
        metric_key: str='val/acc_epoch',
        query_key: str='test_acc'
    ):
        super().__init__(trainer, model, datamodule, cfg)
        assert isinstance(model.mutator, (EAMutator, EvolutionMutator))
        self.mutator = model.mutator
        self.metric_key = metric_key
        self.query_key = query_key
        self.sample_iterations = sample_iterations
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
        try:
            self.mutator.search()

            reals = [info['real_perf'] for encoding, info in self.mutator.vis_dict.items() if 'real_perf' in info]
            proxies = [info['proxy_perf'] for encoding, info in self.mutator.vis_dict.items() if 'proxy_perf' in info]
            tau_visited, p_visited = kendalltau(reals, proxies)

            reals = [info['real_perf'] for info in self.mutator.keep_top_k[self.topk] if 'real_perf' in info]
            proxies = [info['proxy_perf'] for info in self.mutator.keep_top_k[self.topk] if 'proxy_perf' in info]
            tau_topk, p_topk = kendalltau(reals, proxies)

            results = {
                'tau_visited': tau_visited,
                'p_visited': p_visited,
                'tau_topk': tau_topk,
                'p_topk': p_topk,
            } 
            log.info(results)

            with open(
                '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/engine/nb201_ea_search.csv',
                'a') as f:
                #K(subnets), #B(sample_interval), dataset, PretrainedWeights, seed, #valBatch, Tau_visited, P_visited, Tau_topk, P_topk
                valBatch = self.trainer.limit_val_batches
                pw = self.cfg.pretrained_weight
                K = self.cfg.model.num_subnets
                B = self.cfg.model.sample_interval
                seed = self.cfg.seed
                text = f"{K},{B},{dataset},{pw},{seed},{valBatch},{tau_visited},{p_visited},{tau_topk},{p_topk}\n"
                f.write(text)
                log.info(text)
            return results
        except Exception as e:
            self.mutator.save_checkpoint()
            return {'error_info': str(e)}

    
        # self.mutator.start_evolve = True
        # self.mutator.init_population(self.mutator.init_population_mode) # init 
        # self.mutator.crt_epoch = 0
        # while self.mutator.crt_epoch < self.sample_iterations:
        #     print(f"Evolution iteration={self.mutator.crt_epoch}")
        #     if self.mutator.crt_epoch > 0:
        #         self.mutator.evolve()
        #     for j, arch in enumerate(self.mutator.population.values()):
        #         self.mutator.reset_cache_mask(arch['arch'])
        #         self.model.network.sync_mask_for_all_cells(arch['arch'])
        #         if arch.get('metric') is None:
        #             proxy = self.trainer.validate(
        #                 model=self.model, datamodule=self.datamodule, verbose=False)[0][self.metric_key]
        #             arch['metric'] = proxy
        #             real = self.model.network.query_by_key(key=self.query_key, dataset=dataset)
        #             arch_encoding = self.model.network.arch_encoding
        #             if arch_encoding not in self.performance_history:
        #                 self.performance_history[arch_encoding] = {}
        #             self.performance_history[arch_encoding]['proxy'] = proxy
        #             self.performance_history[arch_encoding]['real'] = real
        #     self.mutator.crt_epoch += 1

        # size = np.array([pool['size'] for pool in self.mutator.history.values()])
        # real_metric = np.array([
        #     self.performance_history[pool['arch_code']]['real']
        #         for pool in self.mutator.history.values()
        # ])
        # proxy_metric = np.array([
        #     self.performance_history[pool['arch_code']]['proxy'] 
        #         for pool in self.mutator.history.values()
        # ])
        # # metric = np.array([pool['metric'] for pool in self.mutator.history.values()])
        # indices = np.argsort(size)
        # size, real_metric, proxy_metric = size[indices], real_metric[indices], proxy_metric[indices]
        # epoch = self.mutator.crt_epoch
        # for y_name in [
        #     # 'proxy',
        #     'real'
        # ]:
        #     if y_name == 'proxy':
        #         metric = proxy_metric
        #     elif y_name == 'real':
        #         metric = real_metric
        #     pareto_lists = NonDominatedSorting(np.vstack( (size.reshape(-1), 1/metric.reshape(-1)) ))
        #     pareto_indices = pareto_lists[0] # e.g., [75,  87, 113, 201, 205]
        #     self.mutator.plot_pareto_fronts(
        #         size, metric, pareto_indices, 'model size (MB)', y_name+'_acc',
        #         figname=f'pareto_searchepoch{self.sample_iterations}_{y_name}.pdf'
        #     )
        # tau, p = kendalltau(real_metric, proxy_metric)
        # proxies = []
        # reals = []
        # for arch in self.performance_history:
        #     proxies.append(self.performance_history[arch]['proxy'])
        #     reals.append(self.performance_history[arch]['real'])
        # tau, p = kendalltau(proxies, reals)
        # log.info(f"#valid_batches={self.trainer.limit_val_batches}")
        # log.info(f"Kendall's Tau: {tau}, p-value: {p}")
        # with open(
        #     '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/engine/nb201_results.csv',
        #     'a') as f:
        #     gpus = self.cfg.trainer.gpus
        #     mode = 'hete' if self.cfg.model.is_net_parallel else 'homo'
        #     bs = self.trainer.limit_val_batches
        #     num_nets = self.num_subnets
        #     pw = self.cfg.pretrained_weight
        #     # #GPUs, Hete/Homo, #EvalBatch, #nets, dataset, PretrainedWeights, Tau, P-value
        #     f.write(f"\n{gpus}, {mode}, {bs}, {num_nets}, {dataset}, {pw}, {tau}, {p}")
        # return {'tau': tau, 'p': p}
