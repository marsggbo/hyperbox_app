from glob import glob
import os
import sys

args = sys.argv[1:]
if len(args) == 1:
    path_pattern = args[0]
    path_pattern += '*/*/check*'
else:
    path_pattern = '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/fewshot_search_nb201_*/*/check*'

paths = glob(path_pattern)
print(len(paths), 'paths to evaluate')


i = 0
# for idx, ckpt in enumerate(ckpts[:]):
for idx, path in enumerate(paths[:]):
    gpu_id = i%4
    supernet_mask_path_pattern = f'{path}/*.json'
    ckpts_path_pattern = f'{path}/*sub*.ckpt'
    suffix = path.split('runs/')[1].split('/')[0]
    valid_batches = 1.0
    others = "ipdb_debug=False logger.wandb.offline=True"
    others += f' engine.supernet_mask_path_pattern={supernet_mask_path_pattern}'
    others += f' engine.ckpts_path_pattern={ckpts_path_pattern}'
    cmd = f'''bash ./scripts/fewshot_eval_nb201.sh {gpu_id} {valid_batches} {suffix} "{others}"  & sleep 10'''
    # cmd = template.format(cfg_path, gpu_id, valid_batches, suffix, ckpt, others)
    i += 1
    if i % 8 == 0 and i > 0:
        cmd = cmd.replace('&', '')
    print(f"{cmd}")
    os.system(cmd)
