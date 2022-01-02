exp=$1
# gdas_ccccii
# gdas_nii
# gdas_iran
gpu=$2
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}

others=$3

# CUDA_VISIBLE_DEVICES=1 python -m ipdb run.py \
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=$exp \
datamodule.img_size=[160,160] \
datamodule.center_size=[128,128] \
datamodule.slice_num=32 \
datamodule.num_workers=3 \
datamodule.batch_size=64 \
model/optimizer_cfg=sgd \
model.optimizer_cfg.lr=0.025 \
model.metric_cfg._target_=hyperbox.utils.metrics.Accuracy \
trainer.gpus=$gpuNum \
trainer.accelerator=dp \
$others

# experiment=gdas_ccccii.yaml \
# CUDA_VISIBLE_DEVICES=0 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox_app/medmnist/configs] experiment=gdas_nii.yaml

# CUDA_VISIBLE_DEVICES=2 python -m ipdb run.py hydra.searchpath=[file:///home/comp/18481086/code/hyperbox_app/medmnist/configs] experiment=gdas_iran.yaml