exp=$1
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
suffix=$3
name=${exp}_gpunum${gpuNum}_${suffix}
others=$4
echo $name


CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
experiment=$exp.yaml \
logger.wandb.name=$name \
hydra.job.name=$name \
trainer.accelerator='gpu' \
trainer.strategy=ddp \
trainer.gpus=$gpuNum \
trainer.log_every_n_steps=10 \
trainer.max_epochs=300 \
datamodule=cifar100_datamodule \
datamodule.batch_size=128 \
datamodule.is_customized=False \
model/optimizer_cfg=sgd \
model.optimizer_cfg.lr=0.04 \
+model.is_net_parallel=True \
callbacks.model_checkpoint.save_top_k=2 \
logger.wandb.offline=False \
$others
