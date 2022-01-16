# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperbox.mutables.spaces import InputSpace, OperationSpace, ValueSpace
from hyperbox.mutator.default_mutator import Mutator

__all__ = [
    'DartsWarmupMutator',
]


class DartsWarmupMutator(Mutator):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)
        self.warmup_epoch = 5
        self.current_epoch = 0
        self.choices = nn.ParameterDict()
        for mutable in self.mutables:
            if isinstance(mutable, OperationSpace):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length))
            if isinstance(mutable, ValueSpace):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length))
                mutable.mask = self.choices[mutable.key].data
            elif isinstance(mutable, InputSpace):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.n_candidates))

    @property
    def device(self):
        for v in self.choices.values():
            return v.device

    def random_sample_search(self):
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, OperationSpace):
                gen_index = torch.randint(high=mutable.length, size=(1, ))
                result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = result[mutable.key].detach()
            elif isinstance(mutable, InputSpace):
                if mutable.n_chosen is None:
                    result[mutable.key] = torch.randint(high=2, size=(mutable.n_candidates,)).view(-1).bool()
                else:
                    perm = torch.randperm(mutable.n_candidates)
                    mask = [i in perm[:mutable.n_chosen] for i in range(mutable.n_candidates)]
                    result[mutable.key] = torch.tensor(mask, dtype=torch.bool)  # pylint: disable=not-callable
                mutable.mask = result[mutable.key].detach()
            elif isinstance(mutable, ValueSpace):
                gen_index = torch.randint(high=mutable.length, size=(1, ))
                result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
        return result

    def reset(self, *args, **kwargs):
        """
        Reset the mutator by call the `sample_search` to resample (for search). Stores the result in a local
        variable so that `on_forward_operation_space` and `on_forward_input_space` can use the decision directly.

        Returns
        -------
        None
        """
        if not hasattr(self, 'sample_func'):
            self._cache = self.sample_search(*args, **kwargs)
        else:
            self._cache = self.sample_func(self, *args, **kwargs)
            del self.sample_func
        self._cache = self.check_freeze_mutable(self._cache)

    def sample_search(self, current_epoch=None):
        if current_epoch is None:
            current_epoch = self.current_epoch
        if current_epoch < self.warmup_epoch:
            return self.random_sample_search()
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, OperationSpace):
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1)
                mutable.mask = torch.zeros_like(result[mutable.key])
                mutable.mask[result[mutable.key].cpu().detach().numpy().argmax()] = 1
            elif isinstance(mutable, ValueSpace):
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1)
                mutable.mask.data = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).data
            elif isinstance(mutable, InputSpace):
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1)
                mutable.mask = torch.zeros_like(result[mutable.key])
                mutable.mask[result[mutable.key].cpu().detach().numpy().argmax()] = 1
        return result

    def sample_final(self):
        result = dict()
        edges_max = dict()
        for mutable in self.mutables:
            if isinstance(mutable, (OperationSpace, ValueSpace)):
                max_val, index = torch.max(F.softmax(self.choices[mutable.key], dim=-1), 0)
                edges_max[mutable.key] = max_val
                result[mutable.key] = F.one_hot(index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = torch.zeros_like(result[mutable.key])
                mutable.mask[result[mutable.key].cpu().detach().numpy().argmax()] = 1
        for mutable in self.mutables:
            if isinstance(mutable, InputSpace):
                if mutable.n_chosen is not None:
                    weights = []
                    for src_key in mutable.choose_from:
                        # todo: figure out this issue
                        if src_key not in edges_max:
                            print("InputSpace.NO_KEY in '%s' is weighted 0 when selecting inputs.", mutable.key)
                        weights.append(edges_max.get(src_key, 0.))
                    weights = torch.tensor(weights)  # pylint: disable=not-callable
                    _, topk_edge_indices = torch.topk(weights, mutable.n_chosen)
                    selected_multihot = []
                    for i, src_key in enumerate(mutable.choose_from):
                        if i not in topk_edge_indices and src_key in result:
                            # If an edge is never selected, there is no need to calculate any op on this edge.
                            # This is to eliminate redundant calculation.
                            result[src_key] = torch.zeros_like(result[src_key])
                        selected_multihot.append(i in topk_edge_indices)
                    result[mutable.key] = torch.tensor(selected_multihot, dtype=torch.bool, device=self.device)  # pylint: disable=not-callable
                    mutable.mask = torch.zeros_like(result[mutable.key]) # Todo: 搜索阶段全为1
                    mutable.mask[result[mutable.key].cpu().detach().numpy().argmax()] = 1
                else:
                    result[mutable.key] = torch.ones(mutable.n_candidates, dtype=torch.bool, device=self.device)  # pylint: disable=not-callable
        return result
