
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
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=finetune.yaml \
logger.wandb.name=$name \
hydra.job.name=$name \
datamodule=$datamodule \
trainer.gpus=$gpuNum \
trainer.accelerator=dp \
trainer.max_epochs=100 \
callbacks.model_checkpoint.save_top_k=1 \
model.network_cfg.mask=$mask \
$others
# ++trainer.amp_backend=apex \
# ++trainer.amp_level=o1 \

# example
# bash ./finetune_medmnist.sh organ3dmnist 0 /path/to/mask.json