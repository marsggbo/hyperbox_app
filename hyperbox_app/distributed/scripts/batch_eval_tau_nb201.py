from glob import glob
import os

template = '''
bash ./scripts/eval_tau.sh {} {} {} {} \
{} "{}" &
sleep 10
'''

cfg_path = '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/temp_space/nb201/cfg_c10'
ckpts = glob('/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201*batch*/*/checkpoints/last*')

num = len(ckpts)
i = 0
for idx, ckpt in enumerate(ckpts[:]):
    # print(ckpt)
    # /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201_gpunum1_c10_1net_1batch/2022-04-27_09-20-46/checkpoints/last.ckpt
    gpu_id = i%4
    suffix = ckpt.split('runs/')[1].split('/')[0]
    valid_batches = 1.0
    # suffix = suffix[0] + '_' + suffix[1]
    others = "ipdb_debug=False logger.wandb.offline=True"
    cmd = template.format(cfg_path, gpu_id, valid_batches, suffix, ckpt, others)
    i += 1
    if i % 8 == 0 and i > 0:
        cmd = cmd.replace('&', '')
    print(f"{cmd}")
    # os.system(cmd)
