from glob import glob
import os

template = '''
bash scripts/train_nb201_c10.sh 3 "c10_8net_5batch" "+model.num_subnets=4 +model.sample_interval=1 seed=999 hydra.job.name=debug ipdb_debug=True logger.wandb.offline=True" 
'''

ckpts = glob('/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/*batch*/*/checkpoints/last*')
num = len(ckpts)
i = 0
for idx, ckpt in enumerate(ckpts[:1]):
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
