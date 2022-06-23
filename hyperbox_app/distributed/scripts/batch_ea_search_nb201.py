from glob import glob
import os

template = '''
bash scripts/ea_search_nb201.sh {} {} {} \
{} \
"ipdb_debug=False logger.wandb.offline=True model.mutator_cfg.selection_alg=best model.mutator_cfg.evolution_epochs=20  model.mutator_cfg.topk=0.5" &
sleep 2
'''

# ckpts = glob('/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201*batch*/*/checkpoints/last*')
ckpts = [
    '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201_gpunum1_1net_1batch/2022-05-17_00-11-52/checkpoints/last.ckpt',
    '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201_gpunum1_c10_1net_1batch/2022-05-17_00-14-09/checkpoints/last.ckpt'
]
num = len(ckpts)
i = 2
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
