import copy
import random
from typing import Any, List, Optional, Union

import hydra
import hyperbox_app.medmnist.datamodules as DATASET
import imblearn
import numpy as np
import torch
import torch.nn as nn
from hyperbox.models.base_model import BaseModel
from hyperbox.networks.network_ema import ModelEma
from hyperbox.utils.logger import get_logger
from hyperbox_app.medmnist.utils import getAUC
from omegaconf import DictConfig

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
        is_sync: bool = True,
        is_net_parallel: bool = False,
        **kwargs
    ):
        super().__init__(network_cfg, mutator_cfg, optimizer_cfg,
                         loss_cfg, metric_cfg, scheduler_cfg, **kwargs)
        self.arc_lr = arc_lr
        self.unrolled = unrolled
        self.aux_weight = aux_weight
        self.automatic_optimization = False
        self.is_net_parallel = is_net_parallel
        self.is_sync = is_sync
        self.net_ema = ModelEma(self.network, decay=0.9).eval()

    def on_fit_start(self):
        # self.logger.experiment[0].watch(self.network, log='all', log_freq=100)
        self.sample_search()

    def sample_search(self):
        super().sample_search(self.is_sync, self.is_net_parallel)

    def on_train_epoch_start(self):
        self.y_true_trn = torch.tensor([]).to(self.device)
        self.y_score_trn = torch.tensor([]).to(self.device)
        self.y_true_val = torch.tensor([]).to(self.device)

    def training_step(self, batch: Any, batch_idx: int):
        # debug info
        # self.trainer.accelerator.barrier()
        (trn_X, trn_y) = batch['train']
        (val_X, val_y) = batch['val']
        self.y_true_trn = torch.cat((self.y_true_trn, trn_y), 0)
        self.y_true_val = torch.cat((self.y_true_val, val_y), 0)
        self.weight_optim, self.ctrl_optim = self.optimizers()

        # phase 1. architecture step
        # self.network.eval()
        # self.mutator.train()
        flag1 = 'warmup' not in self.mutator.__class__.__name__.lower()
        flag2 = (not flag1) and self.trainer.current_epoch>self.mutator.warmup_epoch
        if flag1 or flag2:
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
        preds, loss = self._logits_and_loss(trn_X, trn_y, to_aug=True)
        self.y_score_trn = torch.cat((self.y_score_trn, preds), 0)
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(self.network.parameters(), 5.)  # gradient clipping
        self.weight_optim.step()
        self.net_ema.update(self.network)

        # log train metrics
        # preds = torch.argmax(preds, dim=1)
        acc = self.train_metric(preds, trn_y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        if batch_idx % 10 == 0:
            for key, value in self.mutator.choices.items():
                logger.info(f"{key}: {value.detach().softmax(-1)}")
            logger.info(
                f"Train epoch{self.current_epoch} batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": trn_y, 'acc': acc}

    def _logits_and_loss(self, X, y, to_aug, network=None):
        if network is None:
            network = self.network
        output = network(X, to_aug)
        if isinstance(output, tuple):
            output, aux_output = output
            aux_loss = self.criterion(aux_output, y)
        else:
            aux_loss = 0.
        loss = self.criterion(output, y)
        loss = loss + self.aux_weight * aux_loss
        return output, loss

    def _backward(self, val_X, val_y):
        """
        Simple backward with gradient descent
        """
        self.sample_search()
        _, loss = self._logits_and_loss(val_X, val_y, to_aug=True)
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
        _, loss = self._logits_and_loss(val_X, val_y, to_aug=True)
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
        _, loss = self._logits_and_loss(X, y, to_aug=True)
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
            _, loss = self._logits_and_loss(trn_X, trn_y, to_aug=True)
            dalphas.append(torch.autograd.grad(loss, self.mutator.parameters()))

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def training_epoch_end(self, outputs: List[Any]):
        self.y_true_trn = self.y_true_trn.detach().cpu().numpy()
        self.y_score_trn = self.y_score_trn.detach().cpu().numpy()
        logger.info(f'Train class 0 {sum(self.y_true_trn==0)} class 1 {sum(self.y_true_trn==1)} all {len(self.y_true_trn)}')
        self.y_true_val = self.y_true_val.detach().cpu().numpy()
        logger.info(f'Val class 0 {sum(self.y_true_val==0)} class 1 {sum(self.y_true_val==1)} all {len(self.y_true_val)}')
        cls_report = imblearn.metrics.classification_report_imbalanced(self.y_true_trn, self.y_score_trn.argmax(-1), digits=6)
        logger.info(f"Train classification report:\n{cls_report}")
        self.mutator.current_epoch = self.trainer.current_epoch
        acc_epoch = self.trainer.callback_metrics['train/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['train/loss_epoch'].item()
        logger.info(f'Train epoch{self.trainer.current_epoch} acc={acc_epoch:.4f} loss={loss_epoch:.4f}')

        self.net_ema.update_decay(self.current_epoch, self.trainer.max_epochs)

    def on_validation_epoch_start(self):
        self.mutator.current_epoch = self.trainer.current_epoch
        self.y_true = torch.tensor([]).to(self.device)
        self.y_score = torch.tensor([]).to(self.device)
        self.y_score_en = torch.tensor([]).to(self.device)
        self.reset_running_statistics(subset_size=64, subset_batch_size=32)
        # if torch.rand(1) > 0.5:
        #     self.reset_running_statistics(subset_size=64, subset_batch_size=32)
        #     self.eval_net = self.network
        #     logger.info('eval subnet from supernet')
        # else:
        #     self.eval_net = self.net_ema.module.build_subnet(mask=self.mutator._cache).to(self.device)
        #     logger.info('eval subnet from EMA')
        self.eval_net = self.net_ema.module.build_subnet(mask=self.mutator._cache).to(self.device)

    def validation_step(self, batch: Any, batch_idx: int):
        (X, targets) = batch
        preds, loss = self._logits_and_loss(X, targets, to_aug=False, network=self.eval_net)
        self.y_true = torch.cat((self.y_true, targets), 0)
        self.y_score = torch.cat((self.y_score, preds.softmax(dim=-1)), 0)

        # log val metrics
        acc = self.val_metric(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        # if batch_idx % 10 == 0:
        # if True:
        # logger.info(f"Val epoch{self.current_epoch} batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def validation_epoch_end(self, outputs: List[Any]):
        self.y_true = self.y_true.detach().cpu().numpy()
        logger.info(f'class 0 {sum(self.y_true==0)} class 1 {sum(self.y_true==1)} all {len(self.y_true)}')
        self.y_score = self.y_score.detach().cpu().numpy()
        try:
            auc = getAUC(self.y_true, self.y_score, None)
        except:
            auc = -1
        cls_report = imblearn.metrics.classification_report_imbalanced(self.y_true, self.y_score.argmax(-1), digits=6)
        logger.info(f"Validation classification report:\n{cls_report}")
        # logger.info(f'Val epoch{self.trainer.current_epoch} auc={auc:.4f} acc={acc_epoch:.4f} loss={loss_epoch:.4f}')
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        acc_epoch = self.trainer.callback_metrics['val/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['val/loss_epoch'].item()
        logger.info(f'Val epoch{self.trainer.current_epoch} acc={acc_epoch:.4f} loss={loss_epoch:.4f} auc={auc:.4f}')

        # mflops, size = self.arch_size((2, 1, 32, 64, 64), convert=True)
        # logger.info(
        #     f"[rank {self.rank}] current model({self.arch}): {mflops:.4f} MFLOPs, {size:.4f} MB.")
        # logger.info(f"self.mutator._cache: {len(self.mutator._cache)} choices")
        for key, value in self.mutator.choices.items():
            logger.info(f"{key}: {value.detach().softmax(-1)}")

        if self.current_epoch % 1 == 0:
            self.export("mask_epoch_%d.json" % self.current_epoch,
            True, {'val_acc': acc_epoch, 'val_loss': loss_epoch})
        del self.eval_net

    def on_test_epoch_start(self):
        self.y_true = torch.tensor([]).to(self.device)
        self.y_score = torch.tensor([]).to(self.device)
        # self.reset_running_statistics(subset_size=64, subset_batch_size=32)
        # self.eval_net = self.net_ema.module.build_subnet(mask=self.mutator._cache).eval().to(self.device)

    def test_step(self, batch: Any, batch_idx: int):
        (X, targets) = batch
        preds, loss = self._logits_and_loss(X, targets, to_aug=False)
        self.y_true = torch.cat((self.y_true, targets), 0)
        self.y_score = torch.cat((self.y_score, preds.softmax(dim=-1)), 0)

        # log test metrics
        acc = self.test_metric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        if batch_idx % 10 == 0:
            logger.info(f"Test batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def test_epoch_end(self, outputs: List[Any]):
        self.y_true = self.y_true.detach().cpu().numpy()
        self.y_score = self.y_score.detach().cpu().numpy()
        auc = getAUC(self.y_true, self.y_score, None)
        self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        acc = self.trainer.callback_metrics['test/acc'].item()
        loss = self.trainer.callback_metrics['test/loss'].item()
        logger.info(f'Test epoch{self.trainer.current_epoch} acc={acc:.4f} loss={loss:.4f} auc={auc:.4f}')

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_cfg = DictConfig(self.hparams.optimizer_cfg)
        weight_optim = hydra.utils.instantiate(optimizer_cfg, params=self.network.parameters())
        ctrl_optim = torch.optim.Adam(
            self.mutator.parameters(), self.arc_lr, betas=(0.9, 0.999), weight_decay=1.0E-3)
        return weight_optim, ctrl_optim
