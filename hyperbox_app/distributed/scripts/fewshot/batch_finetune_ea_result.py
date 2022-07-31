from glob import glob
import os
import sys
from omegaconf import OmegaConf
from hyperbox.utils.utils import save_arch_to_json
import torch

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--eaPaths", type=str, help="cfg path of previous experiment, e.g., '/path/to/ea_search*/evolution_logs/*tar'")
parser.add_argument("--num_finetune", type=int, default=2, help="number of models to finetune")
parser.add_argument("--others", type=str, help="other cmds", default='')
parser.add_argument("--pt", action='store_true', help="print only")
parser.add_argument("--debug", action='store_true', help="debug")

args = parser.parse_args()

eaPaths = glob(args.eaPaths)
print(eaPaths[0])

num_gpus = torch.cuda.device_count()
print(f"{num_gpus} gpus found")
if num_gpus <= 0:
    sys.exit("No gpus found")

num_cmds = 0
global_idx = 0
for idx, eaPath in enumerate(eaPaths):
    ckpt = torch.load(eaPath)
    topk = ckpt['keep_top_k'][10]
    for idy, info in enumerate(topk[:args.num_finetune]):
        mask = info['arch']
        acc = info['proxy_perf']
        dirname = os.path.dirname(eaPath)
        mask_path = os.path.join(dirname, f"{idy}_{acc:.4f}.json")
        save_arch_to_json(mask, mask_path)
        gpu_id = global_idx % num_gpus
        
        suffix = eaPath.split('runs/')[-1].split('/')[0]
        others = 'logger.wandb.offline=True'
        others += f" {args.others}"

        if args.debug:
            others += ' ipdb_debug=True trainer.fast_dev_run=True'
            suffix = 'debug_' + suffix
        cfgPath = os.path.join(dirname.split('evolution_logs')[0], '.hydra')
        template = f'''bash scripts/fewshot/finetune_ea.sh [{gpu_id}] {suffix} {mask_path} "{others}" &'''
        # template = f'''bash scripts/fewshot/finetune_ea.sh [{gpu_id}] {suffix} {mask_path} {cfgPath} "{others}" &'''
        if (idx+1) % (num_gpus*2) == 0:
            template = template.replace('&', '')

        print(template)
        if not args.pt:
            os.system(template)
        num_cmds += 1
        global_idx += 1
print(f"{num_cmds} cmds generated")