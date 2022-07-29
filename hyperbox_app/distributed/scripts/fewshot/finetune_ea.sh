gpu=$1
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
suffix=$2
mask=$3
# cfgPath=$4
others=$4
name=finetune_${suffix}
# name=debug

echo $name
echo $others

python -m hyperbox.run \
hydra.searchpath=[pkg://hyperbox_app.distributed.configs] \
experiment=finetune.yaml \
hydra.job.name=$name \
logger.wandb.name=$name \
trainer.gpus=$gpu \
$others \
model.network_cfg.mask=$mask \
