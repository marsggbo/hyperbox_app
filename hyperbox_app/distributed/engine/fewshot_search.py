import itertools
import json
import os
import pickle
import random
import types
from copy import deepcopy
from glob import glob
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import skdim
import torch
from hydra.utils import instantiate
from hyperbox.engine.base_engine import BaseEngine
from hyperbox.mutables.spaces import OperationSpace
from hyperbox.mutator import RandomMutator
from hyperbox.utils import logger, utils
from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import hparams_wrapper, load_json, save_arch_to_json
from hyperbox_app.distributed.networks.nasbench201.nasbench201 import \
    NASBench201Network
from hyperbox_app.distributed.networks.nasbenchmb import NASBenchMBNet
from hyperbox_app.distributed.utils.twonn import twonn1, twonn2
from ipdb import set_trace
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from scipy.stats import kendalltau
from skdim.id import ESS
from sklearn.cluster import SpectralClustering

log = get_logger(__name__)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


@hparams_wrapper
class FewshotSearch(BaseEngine):
    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg: DictConfig,
        warmup_epochs: Optional[Union[List[int], int]]=[20, 40],
        finetune_epoch: int=10,
        load_from_parent: bool=False,
        split_criterion: str='ID', # 'ID' or 'grad'
        ID_method: str='lid', # 'lid' or 'ess' or 'mle' or 'twonn'
        split_method: str='spectral_cluster', # 'spectral_cluster' or 'mincut'
        is_single_path: bool=False,
        repeat_num: int=1,
        supernet_masks_path: str=None,
    ):
        super().__init__(trainer, model, datamodule, cfg)
        datamodule_cfg = deepcopy(self.cfg.datamodule)
        datamodule_cfg.batch_size = 128
        datamodule = instantiate(datamodule_cfg)
        datamodule.setup()
        self.dataloader = datamodule.train_dataloader()
        self.mutator = self.model.mutator
        self.supernet_masks_path = supernet_masks_path # e.g., /path/to/*json

    def run(self):
        trainer = self.trainer
        model = self.model
        datamodule = self.datamodule
        config = self.cfg

        if self.supernet_masks_path is None:
            # split search space
            split_mask_history = {}
            supernet_mask = {m.key: torch.ones_like(m.mask) for m in self.mutator.mutables}
            all_supernet_settings = [
                [
                    [trainer, model, supernet_mask, None]
                ],
            ]
            # warmup and split the supernet
            for level, warmup_epoch in enumerate(self.warmup_epochs):
                supernet_settings = all_supernet_settings[level]
                new_supernet_settings = []

                # train all supernets at current level, and split them along the way
                for idx, supernet_setting in enumerate(supernet_settings):
                    flag = f'{level}_{idx}'
                    parent_trainer, parent_model, supernet_masks, best_edge_key = supernet_setting
                    if not isinstance(supernet_masks, list):
                        supernet_masks = [supernet_masks]
                    for supernet_mask in supernet_masks:
                        # warmup
                        trainer, model = self.warmup(
                            parent_trainer, parent_model, datamodule, config,
                            warmup_epoch, supernet_mask)

                        # split current search space (supernet)
                        splitted_supernet_masks, best_infos, best_edge_key = self.split_supernet(
                            trainer, model, datamodule, config,
                            supernet_mask, self.hparams
                        )
                        new_supernet_settings.append([trainer, model, deepcopy(splitted_supernet_masks), best_edge_key])

                if len(new_supernet_settings) > 0:
                    split_mask_history[level] = [deepcopy(s[2]) for s in new_supernet_settings]
                    all_supernet_settings.append(new_supernet_settings)

            level = -1
            save_root_path = os.path.join(os.getcwd(), f'checkpoints')
            if not os.path.exists(save_root_path):
                os.makedirs(save_root_path)
            path = os.path.join(save_root_path, f'split_mask_history.json')
            save_arch_to_json(split_mask_history, path)
            # save supernet masks
            for idx, supernet_setting in enumerate(all_supernet_settings[level]):
                parent_trainer, parent_model, supernet_masks, best_edge_key = supernet_setting
                for idy, supernet_mask in enumerate(supernet_masks):
                    flag = f"level[{level}]-[{idx}-{idy}]-Edge[{best_edge_key}]-subSupernet"
                    mask_path = os.path.join(save_root_path, f"{flag}_mask.json")
                    save_arch_to_json(supernet_mask, mask_path)

            # finetune all supernets
            for idx, supernet_setting in enumerate(all_supernet_settings[level]):
                parent_trainer, parent_model, supernet_masks, best_edge_key = supernet_setting
                for idy, supernet_mask in enumerate(supernet_masks):
                    flag = f"level[{level}]-[{idx}-{idy}]-Edge[{best_edge_key}]-subSupernet"
                    log.info(f"Fintune {flag} with mask {supernet_mask}")
                    trainer, model = self.finetune(
                        parent_trainer, parent_model, datamodule, config,
                        self.finetune_epoch, supernet_mask, self.hparams)
                    results = trainer.callback_metrics

                    ckpt_path = os.path.join(
                        os.getcwd(), f'checkpoints/{flag}_latest.ckpt'
                    )
                    trainer.save_checkpoint(f"{ckpt_path}")
                    log.info(f"Saved {flag} checkpoint to {ckpt_path}")
        else:
            # load supernet masks
            supernet_masks_path = glob(self.supernet_masks_path)
            supernet_masks = [load_json(path) for path in supernet_masks_path]
            
            for idy, path in enumerate(supernet_masks_path):
            # for idy, supernet_mask in enumerate(supernet_masks):
                ckpt_path = path.replace('mask.json', 'latest.ckpt')
                if os.path.exists(ckpt_path):
                    continue
                supernet_mask = load_json(path)
                log.info(f"Fintune [{idy}]-th sub-supernet with mask {supernet_mask}")
                trainer, model = self.finetune(
                    trainer, model, datamodule, config,
                    self.finetune_epoch, supernet_mask, self.hparams)
                results = trainer.callback_metrics
                log.info(f"Fintune Done: [{idy}]-th sub-supernet with mask {supernet_mask} \nresults: {results}")
                
                level = -1
                flag = path.replace('mask.json', 'latest.ckpt')
                ckpt_path = flag
                trainer.save_checkpoint(f"{ckpt_path}")
                log.info(f"Saved [{idy}]-th sub-Supernet checkpoint to {ckpt_path}")
                # flag = f"level[{level}]-[{idx}-{idy}]-Edge[{best_edge_key}]-subSupernet"
                # ckpt_path = os.path.join(
                #     os.getcwd(), f'checkpoints/{flag}_latest_retrain.ckpt'
                # )
                # trainer.save_checkpoint(f"{ckpt_path}")
                # log.info(f"Saved [{idy}]-th sub-Supernet checkpoint to {ckpt_path}")
        return {}

    def warmup(
        self,
        parent_trainer, parent_model, datamodule, config,
        warmup_epoch: int=20, supernet_mask: dict=None):
        '''
        Warmup the model by training for a few epochs.
        '''
        # rebuild a model with the weights of parent_model
        model_cfg = deepcopy(config.model)
        model = instantiate(model_cfg, _recursive_=False)
        model.load_state_dict(parent_model.state_dict())
        # model = deepcopy(parent_model)
        mutator = model.mutator
        mutator.supernet_mask = deepcopy(supernet_mask)

        trainer_cfg = deepcopy(config.trainer)
        trainer_cfg.max_epochs = warmup_epoch
        to_resume = False
        try:
            parent_trainer.save_checkpoint('./temp.ckpt')
            trainer_cfg.resume_from_checkpoint = './temp.ckpt'
            to_resume = True
        except Exception as e:
            pass
        callbacks = parent_trainer.callbacks
        trainer = instantiate(trainer_cfg, callbacks=callbacks,
            logger=parent_trainer.logger, _convert_="partial")
        trainer.fit(model, datamodule)
        if to_resume:
            os.system('rm ./temp.ckpt')
        return trainer, model

    def split_supernet(
        self, trainer, model, datamodule, config,
        supernet_mask: dict, hparams: dict=None
    ):
        '''
        Split the supernet into sub-supernets.
        '''
        base_mask = deepcopy(supernet_mask)
        edge_keys = list(supernet_mask.keys())
        mutator = model.mutator
        is_single_path = hparams.get('is_single_path', False)
        repeat_num = hparams.get('repeat_num', 1)
        split_criterion = hparams.get('split_criterion', 'grad')
        split_method = hparams.get('split_method', 'spectral_cluster')
        ID_method = hparams.get('ID_method', 'lid')

        # best_value = -1e10 if split_method == 'spectral_cluster' else 1e10
        best_value = 0
        best_partition = []
        best_edge_key = None
        best_infos = None
        # enumerate all edges in the supernet
        for edge_key in edge_keys:
            if not base_mask[edge_key].all():
                continue
            num_ops = len(base_mask[edge_key])
            # enumerate all enabled operations of the current edge
            similarity_avg = 0
            for batch_idx, data in enumerate(self.dataloader):
                if batch_idx == repeat_num:
                    break
                
                crt_mask = deepcopy(base_mask)
                for k, v in base_mask.items():
                    if k == edge_key:
                        crt_mask[k] = torch.zeros_like(v)
                    if is_single_path:
                        # sample a single path each time,
                        # each path differs from only one operation in the current edge
                        indices = torch.where(v != 0)[0]
                        gen_index = random.choice(indices)
                        crt_mask[k] = torch.nn.functional.one_hot(gen_index, len(v)).float()
                
                # get gradients
                infos = {}
                for op_idx in range(num_ops):
                    crt_mask[edge_key] = torch.zeros_like(base_mask[edge_key])
                    crt_mask[edge_key][op_idx] = 1
                    crt_mask = {k: v.bool() for k, v in crt_mask.items()}
                    mutator.sample_by_mask(crt_mask)
                    flag = (edge_key, op_idx)
                    if split_criterion == 'grad':
                        grads = calc_grads(model, data, edge_key, crt_mask).cpu().numpy()
                        info = {
                            'mask': crt_mask,
                            'grads': grads,
                            'criterion': grads,
                            'edge_key': edge_key,
                            'op_idx': op_idx,
                        }
                    elif split_criterion == 'ID':
                        IDs = calc_IDs(model.network, data, 2, ID_method)
                        # IDs = calc_IDs(model.network, self.dataloader, 2)
                        info = {
                            'mask': crt_mask,
                            'IDs': IDs,
                            'ID': IDs.mean(0),
                            'criterion': IDs.mean(0),
                            'edge_key': edge_key,
                            'op_idx': op_idx,
                        }
                    infos[flag] = info

                # calculate the similarity between all edges
                if split_criterion == 'grad':
                    similarity = self.calc_similarity(infos, method='cosine') + 1
                elif split_criterion == 'ID':
                    similarity = self.calc_similarity(infos, method='correcoef')
                    similarity = np.nan_to_num(similarity) + 1
                # log.info(f"edge_key={edge_key} similarity={similarity}")
                similarity_avg += similarity
            similarity_avg /= repeat_num
            similarity = similarity_avg - 1
            log.info(f"{edge_key} similarity:\n {similarity}")
            
            if split_method == 'spectral_cluster':
                log.info("Split the supernet into clusters...")
                cluster, cluster_sim_avg = self.gen_cluster(similarity, 2)
                log.info(f"Cluster={cluster} with averaged similarity {cluster_sim_avg:4f}")

                if cluster_sim_avg > best_value:
                    best_value = cluster_sim_avg
                    best_edge_key = edge_key
                    labels = cluster.labels_
                    best_partition = [np.where(labels==i)[0] for i in set(labels)]
                    best_infos = infos
                    log.info(f"Edge {edge_key}: Best cluster={best_partition} with averaged similarity {best_value:4f}")
            elif split_method == 'mincut':
                cut_value, partition = mincut(similarity, 2)
                if cut_value > best_value:
                    best_value = cut_value
                    best_edge_key = edge_key
                    best_partition = partition
                    best_infos = infos
                    log.info(f"Edge {edge_key}: Best cluster={best_partition} with cut value {best_value:4f}")
            elif split_method == 'stoer_wagner':
                G = nx.from_numpy_matrix(similarity)
                cut_value, partition = nx.stoer_wagner(G)
                if cut_value > best_value:
                    best_value = cut_value
                    best_edge_key = edge_key
                    best_partition = partition
                    best_infos = infos
                    log.info(f"Edge {edge_key}: Best cluster={best_partition} with cut value {best_value:4f}")

        supernet_masks = []
        for indices in best_partition:
            crt_mask = deepcopy(base_mask)
            crt_mask[best_edge_key] = torch.zeros_like(base_mask[best_edge_key])
            keys = list(best_infos.keys())
            for idx in range(len(indices)):
                crt_mask[best_edge_key][indices[idx]] = 1
            supernet_masks.append(crt_mask)
        return supernet_masks, best_infos, best_edge_key

    def finetune(
        self,
        parent_trainer, parent_model, datamodule, config,
        finetune_epoch: int, supernet_mask: dict, hparams: dict):
        '''Finetune the model by training for a few epochs.
        '''
        model_cfg = deepcopy(config.model)
        model = instantiate(model_cfg, _recursive_=False)
        load_from_parent = hparams.get('load_from_parent', False)
        if load_from_parent:
            model.load_state_dict(parent_model.state_dict())
        # model = deepcopy(parent_model)
        mutator = model.mutator
        mutator.supernet_mask = deepcopy(supernet_mask)

        trainer_cfg = deepcopy(config.trainer)
        trainer_cfg.max_epochs = hparams.finetune_epoch
        if load_from_parent:
            try:
                parent_trainer.save_checkpoint('./temp.ckpt')
                trainer_cfg.resume_from_checkpoint = './temp.ckpt'
            except Exception as e:
                pass
        callbacks = parent_trainer.callbacks
        trainer = instantiate(trainer_cfg, callbacks=callbacks,
            logger=parent_trainer.logger, _convert_="partial")
        trainer.fit(model, datamodule)
        return trainer, model

    def calc_similarity(self, infos, method='correcoef'):
        '''Calculate the similarity between all edges (infos).
        Args:
            infos: a dict of edge info, with the following keys:
                criterion: the criterion of the edge, e.g., ID, grad
            method: the similarity method, can be 'correcoef' or 'cosine'
        
        Returns:
            similarity: a matrix of similarity between all edges
        '''
        criterions = [info['criterion'] for info in infos.values()]
        if isinstance(criterions[0], torch.Tensor):
            criterions = torch.stack(criterions)
        else:
            criterions = np.vstack(criterions)

        if method == 'correcoef':
            similarity = np.corrcoef(criterions)
        elif method == 'cosine':
            num = criterions.shape[0]
            similarity = np.zeros((num, num))
            for i in range(num):
                for j in range(i, num):
                    similarity[i, j] = np.dot(criterions[i, :], criterions[j, :]) / (
                        np.linalg.norm(criterions[i]) * np.linalg.norm(criterions[j]) + 1e-6
                    )
            tril_indices = np.tril_indices(num, -1)
            similarity[tril_indices] = similarity.T[tril_indices]
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

    def save_pkl(self, data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


# def mincut(dist_avg, split_num): # note: this is not strictly mincut, but it's fine for 201
#     # assert split_num == 2, 'always split into 2 groups for 201 (when using gradient to split)'
#     assert isinstance(dist_avg, np.ndarray)
#     dist_avg = dist_avg - np.tril(dist_avg)
#     best_dist, best_groups, best_edge_score = float('inf'), [], 0
#     for opid1 in range(dist_avg.shape[0]):
#         for opid2 in range(opid1 + 1, dist_avg.shape[0]):
#             group1 = np.array([opid1, opid2]) # always 2
#             group2 = np.setdiff1d(np.array(list(range(dist_avg.shape[0]))), group1)
#             dist = dist_avg[group1[0], group1[1]] + dist_avg[group2[0], group2[1]]
#             if group2.shape[0] > 2:
#                 dist += dist_avg[group2[0], group2[2]] + dist_avg[group2[1], group2[2]]
#             if dist < best_dist:
#                 best_dist = dist
#                 best_groups = [group1, group2]
#                 best_edge_score = dist_avg.sum() - best_dist # dist_avg should be upper-triangular
#     return best_edge_score, best_groups


def mincut(sim_avg, split_num): # note: this is not strictly mincut, but it's fine for 201
    assert split_num == 2, 'always split into 2 groups for 201 (when using gradient to split)'
    assert isinstance(sim_avg, np.ndarray)
    sim_avg = sim_avg - np.tril(sim_avg)
    best_sim, best_groups, best_edge_score = -1*float('inf'), [], 0
    for opid1 in range(sim_avg.shape[0]):
        for opid2 in range(opid1 + 1, sim_avg.shape[0]):
            group1 = np.array([opid1, opid2]) # always 2
            group2 = np.setdiff1d(np.array(list(range(sim_avg.shape[0]))), group1)
            sim = sim_avg[group1[0], group1[1]] + sim_avg[group2[0], group2[1]]
            if group2.shape[0] > 2:
                sim += sim_avg[group2[0], group2[2]] + sim_avg[group2[1], group2[2]]
            if sim > best_sim:
                best_sim = sim
                best_groups = [group1, group2]
                best_edge_score = sim_avg.sum() - best_sim # sim_avg should be upper-triangular
    return best_edge_score, best_groups


def apply_along_axis(function, axis, x):
    return torch.stack([
        function(x_i) for x_i in x
    ], dim=axis)

def lid_term_torch(X, batch, k=20):
    eps = 1e-6
    X = torch.tensor(X).float()
    batch = torch.tensor(batch).float()
    f = lambda v: - k / (torch.sum(torch.log(v / (v[-1]+eps)))+eps)
    distances = torch.cdist(X, batch)
    # print(distances)

    # get the closest k neighbours
    sort_indices = torch.argsort(distances, dim=1)[:, 1:k + 1]
    # print(sort_indices)
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices

    # sorted matrix
    distances_ = distances[tuple(idx)]
    # print(distances_)
    lids = apply_along_axis(f, axis=-1, x=distances_)
    # print(lids)
    return lids.mean()


# def calc_IDs(supernet, dataloader, num_batches=3):
def calc_IDs(supernet, data_batch, num_batches=3, ID_method='lid'):
    supernet = supernet.to(device)
    # device = next(supernet.parameters()).device
    IDs = []
    # for batch_idx, batch in enumerate(dataloader):
    #     id_batch = []
    #     if batch_idx+1 > num_batches:
    #         break
    #     imgs, labels = batch
    id_batch = []
    for imgs, labels in [data_batch]:
        imgs, labels = imgs.to(device), labels.to(device)
        bs = imgs.shape[0]
        with torch.no_grad():
            y = supernet(imgs)
        features = supernet.features
        for idx, feat in enumerate(features):
            if isinstance(feat, torch.Tensor):
                feat = feat.view(bs, -1).detach()
            else:
                feat = feat.reshape(bs, -1)
            try:
                if ID_method == 'lid':
                    _id = lid_term_torch(feat, feat).item()
                else:
                    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-20)
                    feat = feat.cpu().numpy()
                    if ID_method == 'twonn':
                        # _id = twonn2(feat)
                        _id = skdim.id.TwoNN().fit_transform(X=feat)
                    elif ID_method == 'fishers':
                        _id = skdim.id.FisherS().fit_transform(X=feat)
                    elif ID_method == 'ess':
                        _id = skdim.id.ESS().fit_transform(X=feat)
                    elif ID_method == 'mle':
                        _id = skdim.id.MLE().fit_transform(X=feat)
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

def calc_grads(model, data_batch, split_edge_key, crt_mask):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    supernet = model.network.to(device)
    mutator = model.mutator
    device = next(supernet.parameters()).device

    model.configure_optimizers()
    optimizer = model.optimizers()
    if isinstance(optimizer, tuple):
        optimizer = optimizer[0]
    optimizer.zero_grad()
    imgs, labels = data_batch
    imgs, labels = imgs.to(device), labels.to(device)
    bs = imgs.shape[0]
    logits = model(imgs)
    loss = model.criterion(logits, labels)
    loss.backward()
    grads = get_splitted_grads(supernet, mutator, split_edge_key, crt_mask)
    grads = [g.clone().detach().reshape(-1) for g in grads]
    grads = torch.cat(grads, 0)
    return grads    

def get_splitted_grads(model, mutator, split_edge_key, crt_mask):
    params = []
    for name, module in model.named_modules():
        if isinstance(module, OperationSpace):
            if module.key != split_edge_key:
                op_indices = torch.where(crt_mask[module.key]!=0)[0]
                for op_index in op_indices:
                    op = module.candidates[op_index]
                    params += list(op.parameters())
    if hasattr(model, 'classifier'):
        params += list(model.classifier.parameters())
    param_grads = [p.grad for p in params if p.grad is not None]
    return param_grads
