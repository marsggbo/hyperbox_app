cfgPath=$1
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
VB=$3
suffix=$4
name=evalTau_search_${suffix}
pw=$5
others=$6

echo $name

# nb201
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
--config-path=${cfgPath} \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
trainer.limit_val_batches=${VB} \
hydra.job.name=$name \
logger.wandb.name=$name \
pretrained_weight=${pw} \
$others