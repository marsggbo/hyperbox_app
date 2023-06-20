path=$1
mask=$2
weight=$3
others=$4
echo "pretrained_weight='${weight}'"

python -m hyperbox.run \
--config-path=$path \
model.network_cfg.mask=$mask \
"pretrained_weight='${weight}'" \
logger.wandb.offline=True \
ipdb_debug=False \
only_test=True \
debug=False \
$others