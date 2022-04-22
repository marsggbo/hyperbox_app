cfgPath=$1
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
VB=$3
subnetsNum=$4
pw=$5
others=$6

echo val_batch=${VB}_subnetsNum=${subnetsNum}_${others}

# nb201
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
--config-path=${cfgPath} \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
trainer.limit_val_batches=${VB} \
++engine.num_subnets=$subnetsNum \
pretrained_weight=$pw \
$others

# /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201_gpunum1_/2022-04-07_06-17-22/checkpoints/last.ckpt \


# # nbmbnet
# CUDA_VISIBLE_DEVICES=0 python -m hyperbox.run \
# --config-path=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/temp_space/nbmbnet/cfgs \
# hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
# ipdb_debug=False 