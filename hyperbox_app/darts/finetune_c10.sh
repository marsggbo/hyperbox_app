gpu=$1
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
mask=$2
others=$3
name=reproduce_darts_finetune_c10

CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/darts/configs] \
experiment=finetune.yaml \
trainer.gpus=$gpuNum \
trainer.accelerator=dp \
logger.wandb.name=$name \
hydra.job.name=$name \
trainer.max_epochs=600 \
datamodule.is_customized=False \
datamodule.batch_size=96 \
model.network_cfg.mask=$mask \
model.network_cfg.n_layers=20 \
model.network_cfg.channels=36 \
model.network_cfg.auxiliary='cifar' \
model.network_cfg.path_drop_prob=0.2 \
$others

# accelerate the training process
# ++trainer.amp_backend=apex \
# ++trainer.amp_level=o1 \