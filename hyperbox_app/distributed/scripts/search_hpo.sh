exp=$1
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
name=mbv2_gdas_${exp}_gpu${gpuNum}_hpo
others=$3
echo $name


CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
-m hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
hparams_search=search_nb201_hpo \
experiment=$exp.yaml \
logger.wandb.name=$name \
hydra.job.name=$name \
trainer.accelerator='gpu' \
trainer.gpus=$gpuNum \
trainer.log_every_n_steps=10 \
datamodule.is_customized=False \
trainer.max_epochs=5 \
logger.wandb.offline=False \
$others
