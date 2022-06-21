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


class FewshotMutator(RandomMutator):
    def __init__(
        self,
        model,
        *args, **kwargs
    ):
        '''
        '''
        super(FewshotMutator, self).__init__(model)

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

    def sample_final(self):
        return super().sample_search()

if __name__ == '__main__':
    from hyperbox.networks.nasbench201.nasbench201 import NASBench201Network

    def test(net, fm):
        print('\n========test========')
        for i in range(60):
            if fm.idx == fm.num_path:
                print('new round\n=====================')
            fm.reset()
            print(f"{i+1}: {net.arch_encoding} {fm.idx}")
