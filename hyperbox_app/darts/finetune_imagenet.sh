gpu=$1
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
mask=$2
others=$3
name=reproduce_darts_finetune_imagenet

CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/darts/configs] \
experiment=finetune.yaml \
trainer.gpus=$gpuNum \
trainer.accelerator=ddp \
logger.wandb.name=$name \
hydra.job.name=$name \
trainer.max_epochs=250 \
datamodule=imagenet_datamodule \
datamodule.data_dir=~/datasets/imagenet2012 \
datamodule.batch_size=128 \
model.optimizer_cfg.lr=0.1 \
model.network_cfg.mask=$mask \
model.network_cfg.n_layers=14 \
model.network_cfg.channels=36 \
model.network_cfg.auxiliary='imagenet' \
model.network_cfg.path_drop_prob=0.2 \
$others

# accelerate the training process
# ++trainer.amp_backend=apex \
# ++trainer.amp_level=o1 \