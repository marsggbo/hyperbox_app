
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
trainer.gpus=$gpuNum \
trainer.accelerator='gpu' \
trainer.max_epochs=100 \
callbacks.model_checkpoint.save_top_k=1 \
model/network_cfg=dambv2_medmnist \
model.network_cfg.mask=$mask \
++trainer.amp_backend=apex \
++trainer.amp_level=o1 \
$others

# example
# bash ./finetune_medmnist.sh organ3dmnist 0 /path/to/mask.json