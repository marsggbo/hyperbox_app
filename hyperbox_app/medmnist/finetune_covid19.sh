exp=$1
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
name=covid19_finetune_${exp}_gpu${gpuNum}
mask=$3
others=$4

# echo $datamodule
# echo $name
# echo $mask

CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=$exp \
logger.wandb.name=$name \
hydra.job.name=$name \
trainer.gpus=$gpuNum \
trainer.accelerator=dp \
trainer.max_epochs=100 \
callbacks.model_checkpoint.save_top_k=1 \
model.network_cfg.mask=$mask \
$others
# ++trainer.amp_backend=apex \
# ++trainer.amp_level=o1 \

