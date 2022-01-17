from typing import Any, List, Optional, Union

import copy
import random
import hydra
import numpy as np
import torch
import torch.nn as nn
import medmnist
import imblearn
from omegaconf import DictConfig
from kornia.augmentation import RandomMixUp

from hyperbox.utils.logger import get_logger
from hyperbox.models.base_model import BaseModel
from hyperbox_app.medmnist.utils import getAUC
from hyperbox_app.medmnist.losses import MixupLoss, MutualLoss

logger = get_logger(__name__, rank_zero=True)


class DARTSModel(BaseModel):
    def __init__(
        self,
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        scheduler_cfg: Optional[Union[DictConfig, dict]] = None,
        arc_lr: float = 0.001,
        unrolled: bool = False,
        aux_weight: float = 0.4,
        use_mixup: bool = False,
        is_sync: bool = True,
        is_net_parallel: bool = False,
        **kwargs
    ):
        if kwargs.get('datamodule_cfg', None) is not None and \
            'MedMNISTDataModule'.lower() in self.datamodule_cfg._target_.lower():
            self.datamodule_cfg = kwargs.get('datamodule_cfg')
            info = medmnist.INFO[self.datamodule_cfg.data_flag]
            self.c_in = 3 if self.datamodule_cfg.as_rgb else info['n_channels']
            self.num_classes = len(info['label'])
            if network_cfg.get('c_in'):
                network_cfg['c_in'] = self.c_in
            if network_cfg.get('in_channels'):
                network_cfg['in_channels'] = self.c_in
            if network_cfg.get('num_classes'):
                network_cfg['num_classes'] = self.num_classes
        super().__init__(network_cfg, mutator_cfg, optimizer_cfg,
                         loss_cfg, metric_cfg, scheduler_cfg, **kwargs)
        self.arc_lr = arc_lr
        self.unrolled = unrolled
        self.aux_weight = aux_weight
        self.automatic_optimization = False
        self.is_net_parallel = is_net_parallel
        self.is_sync = is_sync
        self.use_mixup = use_mixup
        self.random_mixup = RandomMixUp()

    def on_fit_start(self):
        # self.logger.experiment[0].watch(self.network, log='all', log_freq=100)
        self.sample_search()
        self.dataset_info = self.trainer.datamodule.data_train.info
        self.task = self.dataset_info['task']
        if self.task == 'multi-label, binary-class':
            self.criterion = nn.BCEWithLogitsLoss()
        if self.use_mixup:
            self.criterion = MixupLoss(self.criterion)
        # data_flag = self.trainer.datamodule.data_flag
        # split = self.trainer.datamodule.split
        # root = self.trainer.datamodule.data_dir
        # self.evaluator_train = medmnist.Evaluator(flag=data_flag, split='train', root=root)
        # self.evaluator_val = medmnist.Evaluator(flag=data_flag, split='val', root=root)
        # self.evaluator_test = medmnist.Evaluator(flag=data_flag, split='test', root=root)

    def sample_search(self):
        super().sample_search(self.is_sync, self.is_net_parallel)

    def on_train_epoch_start(self):
        self.y_true_trn = torch.tensor([]).to(self.device)
        self.y_true_val = torch.tensor([]).to(self.device)
        self.y_score_trn = torch.tensor([]).to(self.device)

    def training_step(self, batch: Any, batch_idx: int):
        if self.use_mixup:
            self.criterion.training = True
        self.to_aug = False
        # debug info
        (trn_X, trn_y) = batch['train']
        (val_X, val_y) = batch['val']
        self.y_true_trn = torch.cat((self.y_true_trn, trn_y), 0)
        self.y_true_val = torch.cat((self.y_true_val, val_y), 0)
        self.weight_optim, self.ctrl_optim = self.optimizers()

        # phase 1. architecture step
        # self.network.eval()
        self.mutator.train()
        self.ctrl_optim.zero_grad()
        if self.unrolled:
            self._unrolled_backward(trn_X, trn_y, val_X, val_y)
        else:
            self._backward(val_X, val_y)
        self.ctrl_optim.step()

        # phase 2: child network step
        self.network.train()
        # self.mutator.eval()
        self.weight_optim.zero_grad()
        with torch.no_grad():
            self.sample_search()
        preds, loss = self._logits_and_loss(trn_X, trn_y, to_aug=self.to_aug)
        self.y_score_trn = torch.cat((self.y_score_trn, preds), 0)
        loss_mutual = 0.
        if getattr(self.network, 'num_branches', None):
            num_branches = self.network.num_branches
            num_features = len(self.network.branches[0].features)
            # ensemble_features = []
            for idx in range(num_features):
                en_feat = torch.stack([self.network.branches[i].features[idx] for i in range(num_branches)]).mean(0)
                sub_loss_mutual = 0.
                for i in range(num_branches):
                    sub_loss = MutualLoss('kl')(self.network.branches[i].features[idx], en_feat)
                    sub_loss_mutual += sub_loss
                sub_loss_mutual /= num_branches
                loss_mutual += sub_loss_mutual
                # ensemble_features.append(en_feat)
            loss_mutual /= num_features
            if self.current_epoch < 6:
                loss = loss + 0.1 * loss_mutual
            else:
                loss = loss + 0.8 * loss_mutual
        if self.trainer.world_size > 1:
            preds_en = torch.tensor(self.all_gather(preds.detach())).mean(dim=0)
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(self.network.parameters(), 5.)  # gradient clipping
        self.weight_optim.step()

        # log train metrics
        acc = self.train_metric(preds, trn_y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        acc_ensemble = 0.
        if self.trainer.world_size > 1:
            preds_en = torch.argmax(preds_en, dim=1)
            acc_ensemble = self.train_metric(preds_en, trn_y)
            self.log("train/acc_ensemble", acc_ensemble, on_step=True, on_epoch=True, prog_bar=False)
        if batch_idx % 10 == 0:
            for key, value in self.mutator.choices.items():
                logger.info(f"{key}: {value.detach().softmax(-1)}")
            logger.info(
                f"Train epoch{self.current_epoch} batch{batch_idx}: loss={loss} (mutual={loss_mutual} ce{loss-loss_mutual}), acc={acc}, acc_en={acc_ensemble}")
        return {"loss": loss, "preds": preds.detach(), "targets": trn_y, 'acc': acc}

    def _logits_and_loss(self, X, y, to_aug):
        is_squeeze = False
        if self.task == 'multi-label, binary-class':
            y = y.to(torch.float32)
        else:
            y = y.squeeze().long()
        if self.use_mixup and self.criterion.training:
            if len(X.shape) == 5 and X.shape[1]==1:
                X = X.squeeze(1)
                is_squeeze = True
            X, y = self.random_mixup(X, y)
            if is_squeeze:
                X = X.unsqueeze(1)
        output = self.network(X, to_aug=to_aug)
        if isinstance(output, tuple):
            output, aux_output = output
            aux_loss = self.criterion(aux_output, y)
        else:
            aux_loss = 0.
        output_en = output
        # if self.trainer.world_size > 1:
        #     output_list = self.all_gather(output)
        #     output_en = torch.tensor(output_list).mean(dim=0)
        loss = 0
        if getattr(self.network, 'num_branches', None):
            for i in range(self.network.num_branches):
                loss += self.criterion(output_en[i], y)
            output_en = output_en.mean(0)
        else:
            loss = self.criterion(output_en, y)
        loss = loss + self.aux_weight * aux_loss
        return output_en, loss

    def _backward(self, val_X, val_y):
        """
        Simple backward with gradient descent
        """
        self.sample_search()
        _, loss = self._logits_and_loss(val_X, val_y, to_aug=self.to_aug)
        self.manual_backward(loss)

    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = copy.deepcopy(tuple(self.network.parameters()))

        # do virtual step on training data
        lr = self.optimizer.param_groups[0]["lr"]
        momentum = self.optimizer.param_groups[0]["momentum"]
        weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        self.sample_search()
        _, loss = self._logits_and_loss(val_X, val_y, to_aug=self.to_aug)
        w_model, w_ctrl = tuple(self.network.parameters()), tuple(self.mutator.parameters())
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, trn_X, trn_y)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)

    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay):
        """
        Compute unrolled weights w`
        """
        # don't need zero_grad, using autograd to calculate gradients
        self.sample_search()
        _, loss = self._logits_and_loss(X, y, to_aug=self.to_aug)
        gradients = torch.autograd.grad(loss, self.network.parameters())
        with torch.no_grad():
            for w, g in zip(self.network.parameters(), gradients):
                m = self.optimizer.state[w].get("momentum_buffer", 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.network.parameters(), backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, trn_X, trn_y):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            self.logger.warning(
                "In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.", norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.network.parameters(), dw):
                    p += e * d

            self.sample_search()
            _, loss = self._logits_and_loss(trn_X, trn_y, to_aug=self.to_aug)
            dalphas.append(torch.autograd.grad(loss, self.mutator.parameters()))

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def training_epoch_end(self, outputs: List[Any]):
        self.y_true_trn = self.y_true_trn.detach().cpu().numpy()
        self.y_score_trn = self.y_score_trn.detach().cpu().numpy()
        print('Train class 0', sum(self.y_true_trn==0), 'class 1', sum(self.y_true_trn==1), 'all', len(self.y_true_trn))
        self.y_true_val = self.y_true_val.detach().cpu().numpy()
        print('Val class 0', sum(self.y_true_val==0), 'class 1', sum(self.y_true_val==1), 'all', len(self.y_true_val))
        acc_epoch = self.trainer.callback_metrics['train/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['train/loss_epoch'].item()
        cls_report = imblearn.metrics.classification_report_imbalanced(self.y_true_trn, self.y_score_trn.argmax(-1), digits=6)
        logger.info(f"Train classification report:\n{cls_report}")
        if self.trainer.world_size > 1:
            acc_en_epoch = self.trainer.callback_metrics['train/acc_ensemble_epoch'].item()
            logger.info(f'Train epoch{self.trainer.current_epoch} loss={loss_epoch:.4f} acc={acc_epoch:.4f} acc_en:{acc_en_epoch:.4f}')
        else:
            logger.info(f'Train epoch{self.trainer.current_epoch} loss={loss_epoch:.4f} acc={acc_epoch:.4f}')

    def on_validation_epoch_start(self):
        self.y_true = torch.tensor([]).to(self.device)
        self.y_score = torch.tensor([]).to(self.device)
        self.y_score_en = torch.tensor([]).to(self.device)
        if self.use_mixup:
            self.criterion.training = False
        self.reset_running_statistics(subset_size=64, subset_batch_size=32)

    def validation_step(self, batch: Any, batch_idx: int):
        (X, targets) = batch
        with torch.no_grad():
            preds, loss = self._logits_and_loss(X, targets, to_aug=False)
        self.y_true = torch.cat((self.y_true, targets), 0)
        self.y_score = torch.cat((self.y_score, preds.softmax(dim=-1)), 0)
        if self.trainer.world_size > 1:
            preds_en = torch.tensor(self.all_gather(preds.detach())).mean(dim=0)
            self.y_score_en = torch.cat((self.y_score_en, preds_en.softmax(dim=-1)), 0)

        # log val metrics
        preds = torch.argmax(preds, dim=1)
        acc = self.val_metric(preds, targets)
        if self.trainer.world_size > 1:
            preds_en = torch.argmax(preds_en, dim=1)
            acc_ensemble = self.val_metric(preds_en, targets)
            self.log("val/acc_ensemble", acc_ensemble, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        # if batch_idx % 10 == 0:
        # if True:
        # logger.info(f"Val epoch{self.current_epoch} batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def validation_epoch_end(self, outputs: List[Any]):
        self.y_true = self.y_true.detach().cpu().numpy()
        print('class 0', sum(self.y_true==0), 'class 1', sum(self.y_true==1), 'all', len(self.y_true))
        self.y_score = self.y_score.detach().cpu().numpy()
        cls_report = imblearn.metrics.classification_report_imbalanced(self.y_true, self.y_score.argmax(-1), digits=6)
        logger.info(f"Validation classification report:\n{cls_report}")
        try:
            auc = getAUC(self.y_true, self.y_score, self.task)
        except:
            auc = -1
        acc_epoch = self.trainer.callback_metrics['val/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['val/loss_epoch'].item()
        # logger.info(f'Val epoch{self.trainer.current_epoch} auc={auc:.4f} acc={acc_epoch:.4f} loss={loss_epoch:.4f}')
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        auc_en = 0
        acc_en_epoch = 0
        if self.trainer.world_size > 1:
            self.y_score_en = self.y_score_en.detach().cpu().numpy()
            auc_en = getAUC(self.y_true, self.y_score_en, self.task)
            acc_en_epoch = self.trainer.callback_metrics['val/acc_ensemble_epoch'].item()
            self.log("val/auc_ensemble", auc_en, on_step=False, on_epoch=True, prog_bar=False)
            # logger.info(f'Val epoch{self.trainer.current_epoch} auc_en={auc_en:.4f} acc_en={acc_en_epoch:.4f}')
        logger.info(f'Val epoch{self.trainer.current_epoch} loss={loss_epoch:.4f} auc={auc:.4f} acc={acc_epoch:.4f} auc_en={auc_en:.4f} acc_en={acc_en_epoch:.4f}')

        # mflops, size = self.arch_size((2, 1, 32, 64, 64), convert=True)
        # logger.info(
        #     f"[rank {self.rank}] current model({self.arch}): {mflops:.4f} MFLOPs, {size:.4f} MB.")
        # for key, value in self.mutator._cache.items():
        #     logger.info(f"{key}: {value.detach()}")

        if self.current_epoch % 1 == 0:
            self.export("mask_epoch_%d.json" % self.current_epoch,
            True, {'val_acc': acc_epoch, 'val_loss': loss_epoch})

    def on_test_epoch_start(self):
        if not hasattr(self, 'task'):
            self.dataset_info = self.trainer.datamodule.data_train.info
            self.task = self.dataset_info['task']
            if self.task == 'multi-label, binary-class':
                self.criterion = nn.BCEWithLogitsLoss()
        self.y_true = torch.tensor([]).to(self.device)
        self.y_score = torch.tensor([]).to(self.device)
        self.y_score_en = torch.tensor([]).to(self.device)
        if self.use_mixup:
            self.criterion.training = False
        # self.reset_running_statistics(subset_size=64, subset_batch_size=32)

    def test_step(self, batch: Any, batch_idx: int):
        (X, targets) = batch
        with torch.no_grad():
            preds, loss = self._logits_and_loss(X, targets, to_aug=False)
        if self.trainer.world_size > 1:
            preds_en = torch.tensor(self.all_gather(preds.detach())).mean(dim=0)
            self.y_score_en = torch.cat((self.y_score_en, preds_en.softmax(dim=-1)), 0)
        self.y_true = torch.cat((self.y_true, targets), 0)
        self.y_score = torch.cat((self.y_score, preds.softmax(dim=-1)), 0)

        # log test metrics
        acc = self.test_metric(preds, targets)
        acc_ensemble = 0.
        if self.trainer.world_size > 1:
            preds_en = torch.argmax(preds_en, dim=1)
            acc_ensemble = self.val_metric(preds_en, targets)
            self.log("test/acc_ensemble", acc_ensemble, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        if batch_idx % 10 == 0:
            logger.info(f"Test batch{batch_idx}: loss={loss}, acc={acc} acc_ensemble={acc_ensemble}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def test_epoch_end(self, outputs: List[Any]):
        self.y_true = self.y_true.detach().cpu().numpy()
        self.y_score = self.y_score.detach().cpu().numpy()
        auc = getAUC(self.y_true, self.y_score, self.task)
        acc = self.trainer.callback_metrics['test/acc'].item()
        loss = self.trainer.callback_metrics['test/loss'].item()
        cls_report = imblearn.metrics.classification_report_imbalanced(self.y_true, self.y_score.argmax(-1), digits=6)
        logger.info(f"Test classification report:\n{cls_report}")
        self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        auc_en = 0.
        acc_en_epoch = 0.
        if self.trainer.world_size > 1:
            self.y_score_en = self.y_score_en.detach().cpu().numpy()
            auc_en = getAUC(self.y_true, self.y_score_en, self.task)
            acc_en_epoch = self.trainer.callback_metrics['test/acc_ensemble_epoch'].item()
            self.log("test/auc_ensemble", auc_en, on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/acc_ensemble", acc_en_epoch, on_step=False, on_epoch=True, prog_bar=False)
        logger.info(f'Test epoch{self.trainer.current_epoch} auc={auc:.4f} acc={acc:.4f} auc_en={auc_en:.4f} acc_en={acc_en_epoch:.4f} loss={loss:.4f}')

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_cfg = DictConfig(self.hparams.optimizer_cfg)
        weight_optim = hydra.utils.instantiate(optimizer_cfg, params=self.network.parameters())
        ctrl_optim = torch.optim.Adam(
            self.mutator.parameters(), self.arc_lr, betas=(0.5, 0.999), weight_decay=1.0E-3)
        return weight_optim, ctrl_optim
