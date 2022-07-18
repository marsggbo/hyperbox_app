from glob import glob
import os
import sys

args = sys.argv[1:]
exp_pattern = args[0]

f = lambda path: path.split('/check')[0].split('runs/')[1].replace('splits/','splits_') 
i = 0
for exp_path in glob(exp_pattern):
    if 'cluster' in exp_path or 'ID' in exp_path:
        continue
    json_pattern = f'{exp_path}/*sub*.json'
    ckpt_pattern = f'{exp_path}/*sub*.ckpt'
    jsons = glob(json_pattern)
    ckpts = glob(ckpt_pattern)
    if len(jsons) == 0:
        print(f'{exp_path} has no json')
    if len(jsons) != len(ckpts):
        print("\n==========================================================")
        print(f'{exp_path} has {len(jsons)} json files and {len(ckpts)} ckpt files')
        gpu_id = i%2
        suffix = f(json_pattern)
        others = f"ipdb_debug=False logger.wandb.offline=True hydra.job.name={suffix} ++engine.supernet_masks_path={json_pattern} trainer.limit_val_batches=0"
        cmd = f'''bash ./scripts/fewshot/fewshot_search_nb201.sh {gpu_id} {suffix} "{others}"  & sleep 10'''
        print(f"{cmd}")
        print("\n")
        # os.system(cmd)
        i += 1
print(f"{i} commands to run")