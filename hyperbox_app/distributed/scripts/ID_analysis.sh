cfgPath=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/temp_space/nb201/cfg_c10_ea_new
gpu=$1
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
suffix=$2
pw=$3
others=$4
name=IDAnalysis_${suffix}

echo $name

# nb201
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
--config-path=${cfgPath} \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
hydra.job.name=$name \
logger.wandb.name=$name \
pretrained_weight=${pw} \
model.network_cfg._target_=hyperbox_app.distributed.networks.nasbench201.nasbench201.NASBench201Network \
engine._target_=hyperbox_app.distributed.engine.ID_analysis.IDAnalysis \
+engine.twonn_alg=1 \
+engine.num_subnets=10 \
+engine.search_space_path=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/temp_space/nb201/fixed1000.json \
$others
