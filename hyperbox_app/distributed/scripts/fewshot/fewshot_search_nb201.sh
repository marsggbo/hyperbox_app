gpu=$1
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
suffix=$2
others=$3
name=fewshot_search_${suffix}
# name=debug

echo $name

# nb201
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[pkg://hyperbox_app.distributed.configs] \
experiment=fewshot_search_nb201.yaml \
hydra.job.name=$name \
logger.wandb.name=$name \
trainer.gpus=$gpuNum \
+model.is_net_parallel=True \
$others

# hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
# bash scripts/fewshot_search_nb201.sh 0 nb201 "trainer.fast_dev_run=True ipdb_debug=True"