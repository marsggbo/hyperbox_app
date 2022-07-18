from glob import glob
import os

template = '''
bash ./scripts/ea_search_nbmb_c10.sh {} {} {} \
{} \
"ipdb_debug=False logger.wandb.offline=True model.mutator_cfg.selection_alg=best model.mutator_cfg.evolution_epochs=20 model.mutator_cfg.topk=0.5" &
sleep 2
'''

# ckpts = glob('/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmb*batch*/*/checkpoints/last*')
ckpts = [
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_1net_1batch/2022-05-12_01-22-19/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_1net_1batch/2022-05-12_01-22-21/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_3net_1batch/2022-05-12_01-00-16/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_6net_1batch/2022-05-12_01-22-19/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_6net_1batch/2022-05-12_02-38-38/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_12net_1batch/2022-05-12_01-22-19/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_12net_1batch/2022-05-12_02-38-38/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_1net_1batch/2022-05-12_01-22-19/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_3net_1batch/2022-05-13_05-41-38/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_6net_1batch/2022-05-12_01-22-19/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_originFairNAS_3net_1batch/2022-05-13_05-43-58/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_originFairNAS_3net_1batch/2022-05-13_05-46-10/checkpoints/last.ckpt',

    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_6net_1batch_NoneGrad/2022-05-04_07-11-04/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_6net_1batch_NoneGrad/2022-05-04_10-05-14/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_3net_1batch_NoneGrad/2022-05-04_07-11-04/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_3net_1batch_NoneGrad/2022-05-04_07-21-41/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_12net_1batch_NoneGrad/2022-05-04_07-11-05/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_12net_1batch_NoneGrad/2022-05-04_10-05-14/checkpoints/last.ckpt',
    # '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_originFairNAS_3net_1batch/2022-05-13_05-46-10/checkpoints/last.ckpt',
    '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_1net_1batch/2022-05-16_19-12-18/checkpoints/last.ckpt',
    '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmbnet_gpunum1_1net_1batch/2022-05-16_19-12-30/checkpoints/last.ckpt'
]
num = len(ckpts)
i = 0
for idx, ckpt in enumerate(ckpts[:]):
    # print(ckpt)
    # /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201_gpunum1_c10_1net_1batch/2022-04-27_09-20-46/checkpoints/last.ckpt
    gpu_id = i%4
    suffix = ckpt.split('runs/')[1].split('/')[0]
    valid_batches = 30
    # suffix = suffix[0] + '_' + suffix[1]
    cmd = template.format(gpu_id, suffix, valid_batches, ckpt)
    i += 1
    if i % 8 == 0 and i > 0:
        cmd = cmd.replace('&', '')
    print(f"{cmd}")
    # os.system(cmd)
