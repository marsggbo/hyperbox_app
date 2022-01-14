from typing import Any, List, Optional, Union

import hydra
import torch
from hyperbox.lites.base_lite import HyperboxLite
from hyperbox.models.base_model import BaseModel
from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import DotDict
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.loops import Loop
from torchmetrics import Accuracy


class TrainLoop(Loop):
    def __init__(self, lite, args, model, optimizer, scheduler, dataloader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.args.log_interval = 10
        self.model = model # pytorch_lightning.lite.wrappers._LiteModule
        self.origin_model = model.module # BaseModel
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.dataloader_iter = None
        self.criterion = torch.nn.CrossEntropyLoss()

    @property
    def done(self) -> bool:
        return False

    def reset(self):
        self.dataloader_iter = enumerate(self.dataloader)

    def advance(self, epoch) -> None:
        batch_idx, (data, target) = next(self.dataloader_iter)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        self.lite.backward(loss)
        self.optimizer.step()

        if (batch_idx == 0) or ((batch_idx + 1) % self.args.log_interval == 0):
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(self.dataloader),
                    len(self.dataloader.dataset),
                    100.0 * batch_idx / len(self.dataloader),
                    loss.item(),
                )
            )

        if self.args.dry_run:
            raise StopIteration

    def on_run_end(self):
        self.scheduler.step()
        self.dataloader_iter = None


class TestLoop(Loop):
    def __init__(self, lite, args, model, dataloader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.args.log_interval = 10
        self.model = model
        self.dataloader = dataloader
        self.dataloader_iter = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy().to(lite.device)
        self.test_loss = 0

    @property
    def done(self) -> bool:
        return False

    def reset(self):
        self.dataloader_iter = enumerate(self.dataloader)
        self.test_loss = 0
        self.accuracy.reset()

    def advance(self) -> None:
        _, (data, target) = next(self.dataloader_iter)
        output = self.model(data)
        self.test_loss += self.criterion(output, target)
        self.accuracy(output, target)

        if self.args.dry_run:
            raise StopIteration

    def on_run_end(self):
        test_loss = self.lite.all_gather(self.test_loss).sum() / len(self.dataloader.dataset)

        if self.lite.is_global_zero:
            logger.info(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({self.accuracy.compute():.0f}%)\n")


class MainLoop(Loop):
    def __init__(self, lite, args, model, optimizer, scheduler, train_dataloader, test_dataloader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.epoch = 0
        self.train_loop = TrainLoop(self.lite, self.args, model, optimizer, scheduler, train_dataloader)
        self.test_loop = TestLoop(self.lite, self.args, model, test_dataloader)

    @property
    def done(self) -> bool:
        return self.epoch >= self.args.epochs

    def reset(self):
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.train_loop.run(self.epoch)
        self.test_loop.run()

        if self.args.dry_run:
            raise StopIteration

        self.epoch += 1


class ClassiftLite(HyperboxLite):
    def run(self, args: Optional[Union[DotDict, dict]]=None) -> None:
        default_args = DotDict({
            "epochs": 1000,
            "dry_run": False,
            "log_interval": 10,
            "save_model": True,
        })
        if args is not None and isintsance(args, dict):
            args = DotDict(args)
        else:
            args = default_args

        # datamodule
        datamodule = self.datamodule
        train_dataloader = datamodule.train_dataloader()
        test_dataloader = datamodule.test_dataloader()
        train_dataloader, test_dataloader = self.setup_dataloaders(train_dataloader, test_dataloader)

        # optimizer, scheduler
        model = self.pl_model
        scheduler = None
        if self.hparams.get('scheduler_cfg') is not None:
            optimizer, scheduler = model.configure_optimizers()
        else:
            optimizer = model.configure_optimizers()

        model, optimizer = self.setup(model.network, optimizer)

        MainLoop(self, args, model, optimizer, scheduler, train_dataloader, test_dataloader).run()

        if args.save_model and self.is_global_zero:
            self.save(model.state_dict(), "mnist_cnn.pt")


@hydra.main(config_path="/home/xihe/xinhe/hyperbox/hyperbox/configs/", config_name="lite.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from hyperbox.utils import utils

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    if config.ipdb_debug:
        set_trace()
    
    model_cfg = config.model
    lite_cfg = config.lite
    datamodule_cfg = config.datamodule
    logger_cfg = config.logger
    lite = ClassiftLite(
        lite_cfg=lite_cfg,
        datamodule_cfg=datamodule_cfg,
        logger_cfg=logger_cfg,
        **model_cfg
    )
    lite.run()

if __name__ == "__main__":
    main()
