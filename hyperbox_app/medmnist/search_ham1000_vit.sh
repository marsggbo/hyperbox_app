exp=$1
name=$2
gpu=$3
gpuNum=${gpu//,/}
gpuNum=${#gpuNum}

others=$4

# mpirun -np 2 python run.py \
# mpirun -np 2 python run.py \
# python -m ipdb run.py \
# CUDA_VISIBLE_DEVICES=3 python run.py \
CUDA_VISIBLE_DEVICES=$gpu python -m hyperbox.run \
hydra.searchpath=[file:///home/wangyuxin/xinhe/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=$exp.yaml \
logger.wandb.name=$name \
hydra.job.name=$name \
trainer.accelerator=gpu \
+trainder.strategy=dp \
trainer.gpus=$gpuNum \
trainer.max_epochs=100 \
callbacks.model_checkpoint.monitor='val/auc' \
callbacks.model_checkpoint.save_top_k=2 \
$others

# +model.loss_cfg.weight=[1.,4.] \

# CUDA_VISIBLE_DEVICES=1 bash /home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/run.sh \
# vessel3d vessel3d_ensemble_search \
# "+datamodule.concat_train_val=False +datamodule.use_weighted_sampler=False datamodule.as_rgb=True model.network_cfg.in_channels=3 datamodule.shape_transform=True trainer.max_epochs=100"

# python run.py --config-path=/home/comp/18481086/code/hyperbox/logs/runs/medmnist_finetune_vesselmnist3d/2021-12-17_22-25-17/.hydra/config.yaml hydra.searchpath=[file:///home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/configs]  only_test=True pretrained_weight='/home/comp/18481086/code/hyperbox/logs/runs/medmnist_finetune_vesselmnist3d/2021-12-17_22-25-17/checkpoints/epoch=51_val/acc=0.9219.ckpt'
