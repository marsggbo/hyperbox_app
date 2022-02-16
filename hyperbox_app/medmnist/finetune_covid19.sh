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
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=$exp \
logger.wandb.name=$name \
logger.wandb.offline=False \
hydra.job.name=$name \
datamodule.root_dir=/home/xihe/datasets/COVID19/mosmed/COVID19_1110/studies/ \
+datamodule.use_balanced_batch_sampler=False \
+datamodule.use_weighted_sampler=False \
datamodule.num_workers=3 \
+datamodule.class_weights=[2,1] \
trainer.gpus=$gpuNum \
trainer.accelerator='gpu' \
trainer.max_epochs=100 \
++trainer.reload_dataloaders_every_n_epochs=1 \
++trainer.log_every_n_steps=5 \
callbacks.model_checkpoint.save_top_k=2 \
model/network_cfg=dambv4_covid19 \
+model.network_cfg.first_stride=2 \
model.network_cfg.stride_stages=[2,2,2,1,2,1] \
model.network_cfg.width_stages=[32,48,64,96,160,320] \
model.network_cfg.mask=$mask \
+model.network_cfg.last_channel=1024 \
model.network_cfg.dropout_rate=0 \
+model.network_cfg.input_channel=16 \
+model.network_cfg.ignore_keys=[] \
model.network_cfg.mean=null \
model.network_cfg.std=null \
model.network_cfg.bn_param=[0.1,1e-5] \
model.network_cfg.num_classes=100 \
model.network_cfg.candidate_ops=['3x3_MBConv3','3x3_MBConv4','3x3_MBConv6','5x5_MBConv3','5x5_MBConv4','7x7_MBConv3','7x7_MBConv4','Identity','Zero','Zero'] \
+model.use_mixup=False \
+model.aug_prob=0.8 \
$others
# ++trainer.amp_backend=apex \
# ++trainer.amp_level=o1 \

# example
# model.network_cfg.width_stages=[32,48,64,96,160,320] \ [24,40,80,96,192,256]
# model.network_cfg.width_stages=[32,64,128,256,512,1024] \
# model/optimizer_cfg=adamw \
# model.optimizer_cfg.lr=0.001 \
# model/scheduler_cfg=MultiStepLR \
# model.scheduler_cfg.milestones=[50,80] \
# model.scheduler_cfg.gamma=0.5 \
# ++model.network_cfg.candidate_ops=['3x3_MBConv3SE','3x3_MBConv4SE','5x5_MBConv3SE','7x7_MBConv3SE','Identity'] \
# bash ./finetune_covid19.sh finetune_iran 0 /path/to/mask.json