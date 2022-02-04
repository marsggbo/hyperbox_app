
datamodule=$1
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
name=finetune_${datamodule}_gpu${gpuNum}
mask=$3
others=$4

# echo $datamodule
# echo $name
# echo $mask

CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=finetune.yaml \
logger.wandb.name=$name \
hydra.job.name=$name \
datamodule=$datamodule \
datamodule.batch_size=32 \
datamodule.as_rgb=False \
+datamodule.concat_train_val=True \
+datamodule.use_weighted_sampler=False \
+datamodule.use_balanced_batch_sampler=False \
trainer.gpus=$gpuNum \
trainer.accelerator='gpu' \
trainer.max_epochs=100 \
trainer.log_every_n_steps=10 \
callbacks.model_checkpoint.save_top_k=2 \
model/network_cfg=dambv3_medmnist \
model.network_cfg.mask=$mask \
model/loss_cfg=cross_entropy_labelsmooth \
model/optimizer_cfg=sgd \
model.optimizer_cfg.lr=0.025 \
+model.use_mixup=False \
++trainer.amp_backend=apex \
++trainer.amp_level=o1 \
logger.wandb.offline=False \
$others

# +datamodule.use_weighted_sampler=True \
# model/network_cfg=da_mobile3dnet \
# model/network_cfg=dambv2_medmnist \
# example
# bash ./finetune_medmnist.sh organ3dmnist 0 /path/to/mask.json