# # nb201
# CUDA_VISIBLE_DEVICES=0 python -m hyperbox.run \
# --config-path=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/temp_space/nb201/cfgs \
# hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
# ipdb_debug=False \
# pretrained_weight=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201_gpunum1_/2022-04-07_06-17-22/checkpoints/last.ckpt 

# nbmbnet
CUDA_VISIBLE_DEVICES=0 python -m hyperbox.run \
--config-path=/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/temp_space/nb201/cfgs \
hydra.searchpath=[file:///home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs] \
ipdb_debug=False 