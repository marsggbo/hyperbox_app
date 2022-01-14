gpu=$1
gpuNum=${gpu//,/}
gpuNum=${gpuNum//\'/}
gpuNum=${gpuNum//\"/}
gpuNum=${#gpuNum}

others=$2
name=reproduce_darts_search_gpu$gpuNum
# name=reproduce_darts_search_apexgpu$gpuNum
echo $name
echo $gpuNum

# CUDA_VISIBLE_DEVICES=$gpu python -c 'import torch;print(torch.cuda.device_count())'
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/darts/configs] \
experiment=darts.yaml \
trainer.gpus=1 \
trainer.accelerator='gpu' \
trainer.strategy='ddp' \
logger.wandb.name=$name \
hydra.job.name=$name \
$others

# it can be used to accelerate the training process, but the performance will be degraded
# ++trainer.amp_backend=apex \
# ++trainer.amp_level=o1 \