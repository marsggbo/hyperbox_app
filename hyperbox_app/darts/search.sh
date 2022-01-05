gpu=$1
gpuNum=${gpu//,/}
gpuNum=${gpuNum//\'/}
gpuNum=${gpuNum//\"/}
gpuNum=${#gpuNum}

others=$2
name=reproduce_darts_search_gpu$gpuNum
# name=reproduce_darts_search_apexgpu$gpu
echo $name
echo $gpuNum

# CUDA_VISIBLE_DEVICES=$gpu python -c 'import torch;print(torch.cuda.device_count())'
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/darts/configs] \
experiment=darts.yaml \
trainer.gpus=$gpuNum \
trainer.accelerator=ddp \
logger.wandb.name=$name \
hydra.job.name=$name \
$others

# accelerate the training process
# ++trainer.amp_backend=apex \
# ++trainer.amp_level=o1 \