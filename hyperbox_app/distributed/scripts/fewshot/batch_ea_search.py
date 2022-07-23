from glob import glob
import os
from omegaconf import OmegaConf
import torch

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--expPaths", type=str, help="cfg path of previous experiment, e.g., '/path/to/exp1/configs'")
parser.add_argument("--engineCfg", type=str, default='/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs/engine/ea_search.yaml')
parser.add_argument("--mutatorCfg", type=str, help="mutator config file", default='/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/configs/model/mutator_cfg/EAMutator.yaml')
parser.add_argument("--others", type=str, help="other cmds", default='')
parser.add_argument("--pt", action='store_true', help="print only")
parser.add_argument("--debug", action='store_true', help="debug")

args = parser.parse_args()

# engine_cfg = ['engine._target_=hyperbox_app.distributed.engine.ea_searchNB201.EASearchNB201', 'engine.sample_iterations=1000']
# engine_cfg = OmegaConf.from_dotlist(engine_cfg)
engine_cfg = OmegaConf.load(args.engineCfg)
hydra_cfg = ['hydra.run.dir=logs/runs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}', 'hydra.job.chdir=True']
hydra_cfg = OmegaConf.from_dotlist(hydra_cfg)
mutator_cfg = OmegaConf.load(args.mutatorCfg)

expPaths = glob(args.expPaths)
print(expPaths[0])

num_gpus = torch.cuda.device_count()
print(f"{num_gpus} gpus found")
if num_gpus <= 0:
    sys.exit("No gpus found")

num_cmds = 0
tmp_cfg_path = '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/scripts/fewshot/ea_search_cfgs'
for idx, expPath in enumerate(expPaths):
    cfg = OmegaConf.load(expPath+'/.hydra/config.yaml')
    cfg.model.mutator_cfg = mutator_cfg
    cfg.engine = engine_cfg
    cfg.update(hydra_cfg)

    OmegaConf.save(cfg, tmp_cfg_path+'/config.yaml')

    gpu_id = idx % num_gpus
    suffix = expPath.split('runs/')[-1].split('/2022-')[0]
    lvb = 1.0 # trainer.limit_val_batches
    others = ' logger.wandb.offline=True'
    others = f' ++engine.supernet_mask_path_pattern={expPath}/check*/*mask.json'

    others += args.others

    if args.debug:
        others += ' ipdb_debug=True trainer.fast_dev_run=True'

    template = f'''
    bash scripts/fewshot/ea_search.sh [{gpu_id}] {suffix} {lvb} {tmp_cfg_path}  \
    "{others}" &
    '''
    if (idx+1) % (num_gpus*2) == 0:
        template = template.replace('&', '')

    print(template)
    if not args.pt:
        os.system(template)
    num_cmds += 1
print(f"{num_cmds} cmds generated")