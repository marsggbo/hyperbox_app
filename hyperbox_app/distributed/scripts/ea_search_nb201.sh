cfgPath=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/temp_space/nb201/cfg_c10_ea_new
gpu=$1
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
suffix=$2
VB=$3
pw=$4
others=$5
name=ea_search_${suffix}

echo $name

# nb201
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
--config-path=${cfgPath} \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
hydra.job.name=$name \
logger.wandb.name=$name \
trainer.limit_val_batches=${VB} \
pretrained_weight=${pw} \
$others
