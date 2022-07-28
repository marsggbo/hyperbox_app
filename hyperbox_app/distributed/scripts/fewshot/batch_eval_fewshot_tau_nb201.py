from glob import glob
import os
import sys
from argparse import ArgumentParser

supernet_mask_path_pattern = ''
search_space_path = "/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/networks/nasbench201/top1percent_models.json"
parser = ArgumentParser()
parser.add_argument("--ckpts_path_pattern", type=str)
parser.add_argument("--supernet_mask_path_pattern", type=str)
parser.add_argument("--search_space_path", type=str, default=search_space_path)
parser.add_argument("--other_cmds", type=str, default='')
parser.add_argument("--pt", action='store_true', help="print only")

args = parser.parse_args()
supernet_mask_path_pattern = args.supernet_mask_path_pattern
paths = glob(supernet_mask_path_pattern)
print(len(paths), 'paths to evaluate')


i = 0
# for idx, ckpt in enumerate(ckpts[:]):
for idx, path in enumerate(paths[:]):
    gpu_id = i%4
    supernet_mask_path_pattern = f'{path}/*sub*.json'
    ckpts_path_pattern = f'{path}/*sub*.ckpt'
    suffix = path.split('runs/')[-1].replace('splits/', 'splits_').split('/')[0]
    valid_batches = 1.0
    others = "ipdb_debug=False logger.wandb.offline=True"
    others += f' engine.supernet_mask_path_pattern={supernet_mask_path_pattern}'
    others += f' engine.ckpts_path_pattern={ckpts_path_pattern}'
    if 'c100_' in path:
        others += ' ++model.network_cfg.num_classes=100'
    others += f" {args.other_cmds}"
    cmd = f'''bash ./scripts/fewshot/fewshot_eval_nb201.sh {gpu_id} {valid_batches} {suffix} "{others}"  & sleep 10'''
    # cmd = template.format(cfg_path, gpu_id, valid_batches, suffix, ckpt, others)
    i += 1
    if i % 8 == 0 and i > 0:
        cmd = cmd.replace('&', '')
    print(f"{cmd}")
    if not args.pt:
        os.system(cmd)
print(f'{i} runs')