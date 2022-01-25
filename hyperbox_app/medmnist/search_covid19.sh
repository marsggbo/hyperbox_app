exp=$1
# gdas_ccccii
# gdas_nii
# gdas_iran
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}
name=searchMBV2_${exp}_gpu${gpuNum}

others=$3

CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=$exp \
logger.wandb.name=$name \
hydra.job.name=$name \
+datamodule.use_weighted_sampler=True \
datamodule.img_size=[128,128] \
datamodule.center_size=[96,96] \
datamodule.slice_num=32 \
datamodule.num_workers=3 \
datamodule.batch_size=24 \
model/optimizer_cfg=sgd \
model.optimizer_cfg.lr=0.025 \
model.metric_cfg._target_=hyperbox.utils.metrics.Accuracy \
model/network_cfg=dambv3_covid19 \
trainer.gpus=$gpuNum \
trainer.accelerator='gpu' \
trainer.strategy=null \
trainer.max_epochs=50 \
++trainer.amp_backend=apex \
++trainer.amp_level=o1 \
callbacks.model_checkpoint.monitor='val/auc' \
callbacks.model_checkpoint.save_top_k=1 \
$others

# ++model.network_cfg.candidate_ops=['3x3_MBConv3SE','3x3_MBConv4SE','5x5_MBConv3SE','7x7_MBConv3SE','Identity'] \
# experiment=gdas_ccccii.yaml \
# CUDA_VISIBLE_DEVICES=0 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox_app/medmnist/configs] experiment=gdas_nii.yaml

# CUDA_VISIBLE_DEVICES=2 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox_app/medmnist/configs] experiment=gdas_iran.yaml