exp=$1
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
name=mbv2_gdas_${exp}_gpu${gpuNum}
others=$3
echo $name

CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=$exp.yaml \
logger.wandb.name=$name \
hydra.job.name=$name \
trainer.accelerator='gpu' \
trainer.strategy=null \
trainer.gpus=$gpuNum \
trainer.log_every_n_steps=10 \
+datamodule.concat_train_val=False \
+datamodule.use_weighted_sampler=False \
+datamodule.use_balanced_batch_sampler=False \
datamodule.as_rgb=False \
datamodule.batch_size=24 \
datamodule.is_customized=True \
model/network_cfg=dambv3_medmnist \
model.network_cfg.in_channels=1 \
model/mutator_cfg=darts_warmup_mutator \
model/loss_cfg=cross_entropy_labelsmooth \
trainer.max_epochs=50 \
callbacks.model_checkpoint.monitor='val/auc' \
callbacks.model_checkpoint.save_top_k=2 \
logger.wandb.offline=False \
$others

# +model.loss_cfg.weight=[1.,4.] \

# CUDA_VISIBLE_DEVICES=1 bash /home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/run.sh \
# vessel3d vessel3d_ensemble_search \
# "+datamodule.concat_train_val=False +datamodule.use_weighted_sampler=False datamodule.as_rgb=True model.network_cfg.in_channels=3 datamodule.shape_transform=True trainer.max_epochs=100"

# python run.py --config-path=/home/comp/18481086/code/hyperbox/logs/runs/medmnist_finetune_vesselmnist3d/2021-12-17_22-25-17/.hydra/config.yaml hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs]  only_test=True pretrained_weight='/home/comp/18481086/code/hyperbox/logs/runs/medmnist_finetune_vesselmnist3d/2021-12-17_22-25-17/checkpoints/epoch=51_val/acc=0.9219.ckpt'