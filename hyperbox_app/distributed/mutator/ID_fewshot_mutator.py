import copy
import math
import random
import numpy as np
import skdim
import torch
import torch.nn.functional as F
from hyperbox.mutables.spaces import InputSpace, OperationSpace, ValueSpace
from hyperbox.mutator.random_mutator import RandomMutator
from hyperbox_app.distributed.engine.ID_fewshot import calc_IDs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IDFewshotMutator(RandomMutator):
    def __init__(
        self,
        model,
        *args, **kwargs
    ):
        '''
        '''
        super(IDFewshotMutator, self).__init__(model)

    def sample_search(self, *args, **kwargs):
        result = super().sample_search(*args, **kwargs)
        if getattr(self, 'supernet_mask', None):
            supernet_mask = self.supernet_mask
            for key, val in supernet_mask.items():
                mutable = self[key]
                if not getattr(mutable, 'is_freeze', False):
                    # 1. mutable shoule be not frozen
                    # 2. mutable's mask should be not all ones
                    #    e.g., [1,0,1] indicates the second operation is disabled,
                    #    so we should sample only the first or third operation.
                    candidate_indices = torch.where(val!=0)[0]
                    gen_index = random.choice(candidate_indices)
                    result[key] = F.one_hot(gen_index, num_classes=len(val)).view(-1).bool()
                    self[key].mask = result[key].detach()
        return result

    def sample_search2(self):
        if getattr(self, 'partition_flag', False):
            partition_flag = self.partition_flag
            if partition_flag == 'sub-supernet':
                pass
        if getattr(self, 'precompute_IDs', False):
            key = random.choice(list(self.crt_ID_group.keys()))
            result = self.crt_ID_group[key]['mask']
            self.sample_by_mask(result)
            return result
        max_try = 10        
        while True:
            key = random.choice(list(self.crt_ID_group.keys()))
            result = self.crt_ID_group[key]['mask']
            mutate_key = random.choice(list(result.keys()))
            num = len(result[mutate_key])
            # origin_idx = result[mutate_key].float().argmax()
            # candidate_idx = list(range(0, origin_idx)) + \
            #     list(range(origin_idx + 1, num))
            candidate_idx = list(range(0, num))
            new_idx = random.choice(candidate_idx)
            result[mutate_key] = torch.nn.functional.one_hot(
                torch.tensor(new_idx), num_classes=num).view(-1).bool()
            # result = super().sample_search()
            self.sample_by_mask(result)
            arch = f"{self.model.arch}"
            if arch in self.crt_ID_group:
                return result
            is_visited = False
            candidate_ID_groups = list(range(0, self.crt_ID_group_label)) + \
                list(range(self.crt_ID_group_label + 1, len(self.ID_groups)))
            for ID_group_idx in candidate_ID_groups:
                ID_group = self.ID_groups[ID_group_idx]
                if arch in ID_group:
                    is_visited = True
                    break
            if max_try == 0:
                key = random.choice(list(self.crt_ID_group.keys()))
                return self.crt_ID_group[key]['mask']
            if is_visited:
                max_try -= 1
                continue
            crt_ID_info = self.calc_ID_infos(result)
            crt_ID = crt_ID_info['ID']
            label = self.get_nearest_ID_group_label(crt_ID, self.ID_groups, n_neighbors=3)
            self.ID_groups[label][arch] = crt_ID_info
            if label == self.crt_ID_group_label:
                return result
            max_try -= 1
            # print(max_try)

    def sample_search2(self):
        max_try = 10        
        while True:
            result = super().sample_search()
            self.sample_by_mask(result)
            arch = f"{self.model.arch}"
            if arch in self.crt_ID_group:
                return result
            is_visited = False
            candidate_ID_groups = list(range(0, self.crt_ID_group_label)) + \
                list(range(self.crt_ID_group_label + 1, len(self.ID_groups)))
            for ID_group_idx in candidate_ID_groups:
                ID_group = self.ID_groups[ID_group_idx]
                if arch in ID_group:
                    is_visited = True
                    break
            if max_try == 0:
                key = random.choice(list(self.crt_ID_group.keys()))
                return self.crt_ID_group[key]['mask']
            if is_visited:
                max_try -= 1
                continue
            crt_ID_info = self.calc_ID_infos(result)
            crt_ID = crt_ID_info['ID']
            label = self.get_nearest_ID_group_label(crt_ID, n_neighbors=3)
            self.ID_groups[label][arch] = crt_ID_info
            if label == self.crt_ID_group_label:
                return result
            max_try -= 1
            print(max_try)

    def sample_final(self):
        return super().sample_search()

    def get_nearest_ID_group_label(self, crt_ID, ID_groups, n_neighbors=3):
        best_label = 0
        best_sim = -2
        for label, ID_group in ID_groups.items():
            sim = 0
            if n_neighbors > len(ID_group):
                indices = list(range(len(ID_group)))
            else:
                indices = np.random.choice(len(ID_group), n_neighbors, replace=False)
            keys = list(ID_group.keys())
            IDs = np.array([ID_group[keys[i]]['ID'] for i in indices])
            sim = np.corrcoef(crt_ID, IDs)[0, 1:].mean()
            # for ID in IDs:
            #     sim += np.corrcoef(IDs, crt_ID)[0, 1]
            # sim /= n_neighbors
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label

    def calc_ID_infos(self, arch_mask: dict):
        # self.sample_by_mask(arch_mask)
        ID_info = {
                'arch': f"{self.model.arch}",
                'mask': arch_mask,
                'IDs': None,
                'ID': None,
                'key': None,
                'proxy_acc': None,
                'real_acc': None
        }

        IDs = calc_IDs(self.model, self.test_loader, 2)
        ID_info['IDs'] = IDs
        ID_info['ID'] = IDs.mean(0)
        return ID_info


if __name__ == '__main__':
    from hyperbox.networks.nasbench201.nasbench201 import NASBench201Network

    def test(net, fm):
        print('\n========test========')
        for i in range(60):
            if fm.idx == fm.num_path:
                print('new round\n=====================')
            fm.reset()
            print(f"{i+1}: {net.arch_encoding} {fm.idx}")
