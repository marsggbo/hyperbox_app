exp=$1
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
name=covid19_${exp}_gpu${gpuNum}
mask=$3
others=$4

# echo $datamodule
# echo $name
# echo $mask

CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=$exp \
logger.wandb.name=$name \
hydra.job.name=$name \
+datamodule.use_weighted_sampler=False \
trainer.gpus=$gpuNum \
trainer.accelerator='gpu' \
trainer.max_epochs=100 \
callbacks.model_checkpoint.save_top_k=2 \
model/loss_cfg=cross_entropy_labelsmooth \
model/network_cfg=dambv3_covid19 \
+model.network_cfg.first_stride=1 \
model.network_cfg.stride_stages=[2,2,2,2,2,1] \
model.network_cfg.mask=$mask \
+model.network_cfg.input_channel=16 \
+model.network_cfg.ignore_keys=['gauNoise','erase']
model/optimizer_cfg=adamw \
model.optimizer_cfg.lr=0.001 \
model/scheduler_cfg=MultiStepLR \
model.scheduler_cfg.milestones=[50,80] \
model.scheduler_cfg.gamma=0.5 \
+model.use_mixup=False \
$others
# ++trainer.amp_backend=apex \
# ++trainer.amp_level=o1 \

# example
# ++model.network_cfg.candidate_ops=['3x3_MBConv3SE','3x3_MBConv4SE','5x5_MBConv3SE','7x7_MBConv3SE','Identity'] \
# bash ./finetune_covid19.sh finetune_iran 0 /path/to/mask.json