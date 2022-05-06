from glob import glob
import os

template = '''
bash ./scripts/ea_search_nbmb_c10.sh {} {} {} \
{} \
"ipdb_debug=False logger.wandb.offline=True model.mutator_cfg.selection_alg=best model.mutator_cfg.evolution_epochs=30" &
sleep 2
'''

ckpts = glob('/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nbmb*batch*/*/checkpoints/last*')
num = len(ckpts)
i = 0
for idx, ckpt in enumerate(ckpts[:]):
    # print(ckpt)
    # /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201_gpunum1_c10_1net_1batch/2022-04-27_09-20-46/checkpoints/last.ckpt
    gpu_id = i%4
    suffix = ckpt.split('runs/')[1].split('/')[0]
    valid_batches = 50
    # suffix = suffix[0] + '_' + suffix[1]
    cmd = template.format(gpu_id, suffix, valid_batches, ckpt)
    i += 1
    print(f"{cmd}\n{idx+1}/{num}")
    os.system(cmd)
