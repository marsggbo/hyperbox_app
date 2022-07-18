from glob import glob
import os
import sys
import torch

num_gpus = torch.cuda.device_count()

args = sys.argv[1:]
masks_pattern = args[0] # 'path/to/*.json'
masks2finetune = glob(masks_pattern)


i = 0
for idx, mask in enumerate(masks2finetune):
    gpu_id = i%num_gpus
    suffix = f"net{idx}"
    others = "ipdb_debug=False logger.wandb.offline=True trainer.strategy=null trainer.limit_val_batches=1.0"
    cmd = f'''bash ./scripts/fewshot/finetune.sh {gpu_id} {suffix} {mask} "{others}"  & sleep 10'''
    i += 1
    if i % (num_gpus * 2) == 0 and i > 0:
        cmd = cmd.replace('&', '')
    print(f"{cmd}")
    os.system(cmd)
print(f'{i} commands to run')