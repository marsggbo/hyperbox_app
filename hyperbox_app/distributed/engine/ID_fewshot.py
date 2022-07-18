from typing import List, Optional, Union
import json
import os
from glob import glob
from copy import deepcopy
import pickle
import itertools

import numpy as np
import skdim
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scipy.stats import kendalltau
from skdim.id import ESS
from sklearn.cluster import SpectralClustering
from ipdb import set_trace

from hyperbox.mutables.spaces import OperationSpace
from hyperbox.engine.base_engine import BaseEngine
from hyperbox.mutator import RandomMutator
from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import load_json, hparams_wrapper
from hyperbox_app.distributed.networks.nasbench201.nasbench201 import \
    NASBench201Network
from hyperbox_app.distributed.networks.nasbenchmb import NASBenchMBNet
from hyperbox_app.distributed.utils.twonn import twonn1, twonn2
from hyperbox.utils import utils, logger

log = get_logger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


@hparams_wrapper
class IDFewshot(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        ckpt_weight: str,
        ckpt_ea: str,
        partition_flag: str='sub-supernet',
        partition_metric: str='ID',
        num_IDs: int=100,
        ID_groups_path: str=None,
        precompute_IDs: bool=False,
    ):
        super().__init__(trainer, model, datamodule, cfg)
        self.mutator = self.model.mutator

        log.info(f"Calc IDs...")
        datamodule_cfg = deepcopy(self.cfg.datamodule)
        datamodule_cfg.batch_size = 16
        datamodule = instantiate(datamodule_cfg)
        datamodule.setup()
        self.dataloader = datamodule.train_dataloader()
        self.ID_groups = self.get_ID_groups(
            partition_flag, partition_metric, ckpt_ea, ckpt_weight, num_IDs, ID_groups_path, precompute_IDs,
        )

    def get_ID_groups(
        self,
        partition_flag: str,
        partition_metric: str,
        ckpt_ea: str,
        ckpt_weight: str,
        num_IDs: int=100,
        ID_groups_path: str=None,
        precompute_IDs: bool=False,
    ) -> dict:
        '''Get ID groups by partitioning search space
        Args:
            partition_flag: 'sub-supernet', 'subnet', 'spectral'
        '''        
        if partition_flag == 'sub-supernet':
            self.ID_groups = self.enumerate_sub_supernets(partition_metric)
        elif partition_flag == 'subnet':
            if ID_groups_path is not None:
                with open(ID_groups_path, "rb") as f:
                    self.ID_groups = pickle.load(f)
            else:
                self.ID_infos = self.calc_ID_infos(
                    self.model.network,
                    self.mutator,
                    ckpt_ea,
                    ckpt_weight,
                    dataloader=self.dataloader,
                    topk=-1,
                    num_IDs=num_IDs
                )
                self.IDs = np.array([val['ID'] for k, val in self.ID_infos.items()])

                log.info("Calc similarity...")
                self.similarity = self.calc_similarity(self.ID_infos)

                log.info("Generate cluster...")
                self.cluster, self.cluster_sim_avg = self.gen_cluster(self.similarity)
                log.info(f"Cluster={self.cluster} with averaged similarity {self.cluster_sim_avg:4f}")
                self.labels = self.cluster.labels_
                self.u_labels = set(self.labels)

                log.info("Cluster ID groups ...")
                self.ID_groups = {}
                self.crt_ID_group_label = 0
                for label in self.u_labels:
                    indices = np.where(self.labels == label)[0]
                    # crt_ID_group = {idx: self.ID_infos[indices[idx]] for idx in range(len(indices))}
                    crt_ID_group = {}
                    for idx in range(len(indices)):
                        id_info = self.ID_infos[indices[idx]]
                        arch = id_info['arch']
                        crt_ID_group[f"{arch}"] = id_info
                    self.ID_groups[label] = crt_ID_group
                with open(f"precompute_IDs.pkl", "wb") as f:
                    pickle.dump(self.ID_groups, f)
            if precompute_IDs:
                self.mutator.precompute_IDs = precompute_IDs
                try:
                    self.ID_groups = self.enumarate_all_subnets(self.ID_groups)
                except (Exception, KeyboardInterrupt) as e:
                    log.error(f"Failed to enumerate all subnets: {e}")
                    raise e
                finally:
                    with open(f"precompute_IDs.pkl", "wb") as f:
                        pickle.dump(self.ID_groups, f)
        return self.ID_groups

    def enumerate_sub_supernets(self, partition_metric: str='ID') -> dict:
        """
        Enumerate all subnets and supernets of each ID group.
        """
        infos = {}
        base_mask = {}
        for m in self.mutator.mutables:
            base_mask[m.key] = torch.ones_like(m.mask)
        keys = list(base_mask.keys())
        for key in keys:
            num = len(base_mask[key])
            for i in range(num):
                mask = deepcopy(base_mask)
                mask[key][i] = 0
                mask = {k: v.bool() for k, v in mask.items()}
                self.mutator.sample_by_mask(mask)
                flag = (key, i)
                if partition_metric == 'ID':
                    info = {
                            'arch': flag,
                            'mask': mask,
                            'IDs': None,
                            'ID': None,
                            'key': key,
                            'partition_metric': partition_metric,
                            'op_idx': i,
                    }
                    IDs = calc_IDs(self.model.network, self.dataloader, 20)
                    info['IDs'] = IDs
                    info['ID'] = IDs.mean(0)
                elif partition_metric == 'gradient':
                    info = {
                            'arch': flag,
                            'mask': mask,
                            'grads': None,
                            'partition_metric': partition_metric,
                            'key': key,
                            'op_idx': i,
                    }
                    grads = calc_grads(self.model, self.dataloader, flag, 20)
                infos[flag] = info

        self.infos = infos
        log.info("Calc similarity...")
        self.similarity = self.calc_similarity(self.infos)

        log.info("Generate cluster...")
        self.cluster, self.cluster_sim_avg = self.gen_cluster(self.similarity)
        log.info(f"Cluster={self.cluster} with averaged similarity {self.cluster_sim_avg:4f}")
        self.labels = self.cluster.labels_
        self.u_labels = set(self.labels)

        log.info("Cluster groups ...")
        self.groups = {}
        self.crt_ID_group_label = 0
        for label in self.u_labels:
            indices = np.where(self.labels == label)[0]
            # crt_ID_group = {idx: self.infos[indices[idx]] for idx in range(len(indices))}
            crt_ID_group = {}
            for idx in range(len(indices)):
                info = self.infos[indices[idx]]
                arch = info['arch']
                crt_ID_group[f"{arch}"] = info
            self.groups[label] = crt_ID_group
        with open(f"groups.pkl", "wb") as f:
            pickle.dump(self.groups, f)

        return self.groups

    def enumarate_all_subnets(self, ID_groups: dict) -> dict:
        '''
        Enumerate IDs of all subnets in the model.
        '''
        subnets_list = []
        for m in self.mutator.mutables:
            subnets_list.append(list(range(m.length)))
        subsets_list = list(itertools.product(*subnets_list))
        num = sum([len(ID_groups[i]) for i in ID_groups])
        if num == len(subsets_list):
            log.info(f"All subnets are enumerated.")
            return ID_groups
        # np.random.shuffle(subsets_list)
        for idx, subnet_enc in enumerate(subsets_list):
            if idx % 1000 == 0:
                log.info(f"{idx}/{len(subsets_list)}")
                with open(f"precompute_IDs.pkl", "wb") as f:
                    pickle.dump(ID_groups, f)
            mask = {}
            for idx, m in enumerate(self.mutator.mutables):
                mask[m.key] = torch.nn.functional.one_hot(
                    torch.tensor(subnet_enc[idx]), num_classes=m.length).view(-1).bool()
            self.mutator.sample_by_mask(mask)
            arch = f"{self.model.arch}"
            is_visited = False
            for ID_group_idx in ID_groups.keys():
                ID_group = ID_groups[ID_group_idx]
                if arch in ID_group:
                    is_visited = True
                    break
            if not is_visited:
                ID_info = {
                        'arch': f"{self.model.network.arch}",
                        'mask': mask,
                        'IDs': None,
                        'ID': None,
                        'key': None,
                        'proxy_acc': None,
                        'real_acc': None
                }

                IDs = calc_IDs(self.model.network, self.dataloader, 2)
                ID_info['IDs'] = IDs
                ID_info['ID'] = IDs.mean(0)
                label = self.mutator.get_nearest_ID_group_label(ID_info['ID'], ID_groups, n_neighbors=3)
                ID_groups[label][arch] = ID_info

        return ID_groups

    def run(self):
        origin_trainer = self.trainer
        model = self.model
        datamodule = self.datamodule
        config = self.cfg
        self.mutator.ID_groups = self.ID_groups
        self.mutator.dataloader = self.dataloader
        for label, ID_group in self.ID_groups.items():
            self.mutator.crt_ID_group_label = label
            self.mutator.crt_ID_group = self.ID_groups[label]
            trainer_cfg = deepcopy(config.trainer)
            trainer_cfg.max_epochs = origin_trainer.max_epochs // len(self.ID_groups)
            # trainer_cfg.limit_train_batches = 1
            trainer = instantiate(trainer_cfg, callbacks=origin_trainer.callbacks,
                logger=origin_trainer.logger, _convert_="partial")
            # Train the model
            log.info(f"Starting training for the {label}-th group!")
            trainer.fit(model=model, datamodule=datamodule)
            result = trainer.callback_metrics
            path = os.path.join(
                os.getcwd(), f'checkpoints/ID_group{label}_latest.ckpt'
            )
            trainer.save_checkpoint(f"{path}")
            log.info(f"Saved {label}-th ID group checkpoint to {path}")
            log.info(f'result={result}')

            # # Evaluate model on test set, using the best model achieved during training
            # if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
            #     log.info("Starting testing!")
            #     ckpt_path = config.trainer.get('ckpt_path') or "best"
            #     try:
            #         trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            #     except:
            #         trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
            #     result.update(trainer.callback_metrics)
        with open(f"checkpoints/ID_fewshot_result.pkl", "wb") as f:
            pickle.dump(self.ID_groups, f)
        return {}

    def calc_ID_infos(
        self,
        supernet,
        mutator,
        ckpt_ea,
        ckpt_weight,
        dataloader,
        topk=-1,
        flag: str='vis', # 'keep_top_k' or 'vis'
        num_IDs: int=100
    ) -> dict:
        '''
        Args:
            ckpt_ea: checkpoint of evolution results
            ckpt_weight: checkpoint of model
        
        Returns:
            {
                0: {
                    'arch': '021021',
                    'mask': {},
                    'IDs': [], # IDs of multiple calculations
                    'ID': [20, 15, 30], # mean of IDs
                    'key': f"{model_idx}_{supernet.arch}_proxyAcc{acc:.2f}_realAcc{real_acc:.2f}",
                    'proxy_acc': 80.25,
                    'real_acc': 87.26,
                } ,
                1: {},
                ...
            }
        '''
        ID_infos = {}
        evolution_logs = torch.load(ckpt_ea)
        arch_keys = list(evolution_logs['vis_dict'].keys())
        if topk==-1:
            infos = evolution_logs['vis_dict']
        else:
            assert topk in evolution_logs['keep_top_k'], f'{topk} not in keep_top_k'
            infos = evolution_logs['keep_top_k'][topk]
        np.random.shuffle(arch_keys)
        num_IDs = min(num_IDs, len(infos))
        supernet.load_from_ckpt(ckpt_weight)
        supernet = supernet.to(device)
        mutator = mutator.to(device)
        supernet.eval()
        for model_idx in range(num_IDs):
            if flag == 'vis':
                # vis_dict
                mask = infos[arch_keys[model_idx]]['arch']
                acc = infos[arch_keys[model_idx]]['proxy_perf'] * 100
            else:
                # keep_top_k
                mask = infos[model_idx]['arch']
                acc = infos[model_idx]['proxy_perf']*100
            mutator.sample_by_mask(mask)
            real_acc = supernet.query_by_key() # only for NASBench search space
            key = f"{model_idx}_{supernet.arch}_acc{acc:.2f}"
            log.info(key)
            ID_infos[model_idx] = {
                'arch': f"{supernet.arch}",
                'mask': mask,
                'IDs': None,
                'ID': None,
                'key': key,
                'proxy_acc': acc,
                'real_acc': real_acc
            }
            IDs = calc_IDs(supernet, dataloader)
            ID_infos[model_idx]['IDs'] = IDs
            ID_infos[model_idx]['acc'] = acc
            ID_infos[model_idx]['ID'] = IDs.mean(0)
        return ID_infos

    def calc_similarity(self, ID_infos: dict, sim='corrcoef') -> np.array:
        '''
        Args:
            ID_infos: dict (returned value of `calc_ID_infos`)
            sim: 'corrcoef' or 'cosine' or 'covariance'
        '''
        keys = list(ID_infos.keys())
        IDs = []
        for idx, key in enumerate(keys):
            ID = np.array(ID_infos[key]['ID'])
            IDs.append(ID)
        IDs = np.vstack(IDs)
        if sim == 'corrcoef':
            similarity = np.corrcoef(IDs)
        elif sim == 'cosine':
            num = IDs.shape[0]
            similarity = np.random.rand(num, num)
            for i in range(num):
                for j in range(i, num):
                    x = IDs[i, :]
                    y = IDs[j, :]
                    similarity[i, j] = np.sum(x * y, axis=-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 0.000001)
            tril_indices = np.tril_indices(num, -1)
            similarity[tril_indices] = similarity.T[tril_indices]
        elif sim == 'covariance':
            IDs = (IDs - IDs.min()) / (IDs.max() - IDs.min() + 1e-20)
            similarity = np.cov(IDs)
        return similarity

    def gen_cluster(self, similarity: np.array, n_clusters: int=None):
        x = similarity
        x = np.nan_to_num(x) + 1 
        best_cluster = None
        best_n_clusters = 0
        best_sim = 0
        best_labels = None
        if n_clusters is not None:
            n_cluster_list = [n_clusters]
        else:
            n_cluster_list = list(range(2, 10))
        for n_clusters in n_cluster_list:
            cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', affinity='precomputed', random_state=0)
            cluster.fit_predict(x)
            labels = cluster.labels_
            u_labels = np.unique(labels)
            sim_avg = 0
            for label in u_labels:
                mask = labels == label
                sim = np.nan_to_num(similarity)[mask][:, mask]
                # log.info(f"label={label}: {sum(labels==label)} {sim.mean()}")
                sim_avg += sim.mean()
            sim_avg /= len(u_labels)
            if sim_avg > best_sim:
                best_sim = sim_avg
                best_cluster = cluster
                best_n_cluster = n_clusters
                best_labels = labels
            log.info(f"n_clusters={n_clusters} avg similarity={sim_avg}")
        log.info(f"Best settings: {best_cluster} with sim {best_sim}")
        # return best_cluster, best_n_clusters, best_sim, best_labels
        return best_cluster, best_sim


def calc_IDs(supernet, dataloader, num_batches=3):
    device = next(supernet.parameters()).device
    IDs = []
    for batch_idx, batch in enumerate(dataloader):
        id_batch = []
        if batch_idx+1 > num_batches:
            break
        imgs, labels = batch
        bs = imgs.shape[0]
        with torch.no_grad():
            y = supernet(imgs.to(device))
        features = supernet.features
        for idx, feat in enumerate(features):
            if isinstance(feat, torch.Tensor):
                feat = feat.view(bs, -1).detach().cpu().numpy()
            else:
                feat = feat.reshape(bs, -1)
            try:
                feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-20)
                # _id = twonn2(feat)
                # _id = skdim.id.TwoNN().fit_transform(X=feat)
                # _id = skdim.id.FisherS().fit_transform(X=feat)
                _id = skdim.id.ESS().fit_transform(X=feat)
                # _id = skdim.id.MLE().fit_transform(X=feat)
                if _id is np.nan:
                    set_trace()
                    log.info(_id)
            except Exception as e:
                set_trace()
                log.info(idx, str(e))
                log.info(feat.shape, feat.max(), feat.mean(), feat.min())
            id_batch.append(_id)
        # id_batch_1 = np.array(id_batch)[1:]-np.array(id_batch)[:-1]
        # id_batch_2 = np.array(id_batch_1)[1:]-np.array(id_batch_1)[:-1]
        # id_batch = id_batch_1.tolist() + id_batch_2.tolist()
        IDs.append(id_batch)
    IDs = np.vstack(IDs)
    return IDs
