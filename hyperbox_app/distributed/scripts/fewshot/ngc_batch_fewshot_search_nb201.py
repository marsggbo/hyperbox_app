import itertools
import os
import sys
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch

global_pool_path_ = '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/networks/nasbench201/all_mask_IDs_dbscan.pt'

parser = ArgumentParser()
parser.add_argument("--split_method", nargs="+", type=str, default=["mincut"])
parser.add_argument("--similarity_method", nargs="+", type=str, default=["cosine"])
parser.add_argument("--split_criterion", nargs="+", type=str, default=["ID", "ID"])
parser.add_argument("--split_num", type=str, default="[2]")
parser.add_argument("--is_single_path", nargs="+", type=int, default=[0, 0])
parser.add_argument("--to_sample_similar", nargs="+", type=int, default=[0])
parser.add_argument("--load_from_parent", nargs="+", type=int, default=[1])
parser.add_argument("--global_pool_path", type=str, default=None)
parser.add_argument("--ID_method", type=str, default='lid')
parser.add_argument("--warmup_epochs", nargs="+", type=str, default=["[50,75,90,100]"])
parser.add_argument("--finetune_epoch", type=int, default=50)
parser.add_argument("--debug", action='store_true')
parser.add_argument("--supernet_masks_path", type=str, default=None)
parser.add_argument("--network_cfg", type=str, default='nb201') # mbv3, nbmb, spos
parser.add_argument("--datamodule", type=str, default='cifar10_datamodule') # cifar100_datamodule
parser.add_argument("--other_cmds", type=str, default=None)
parser.add_argument("--pt", action='store_true', help="print only")
args = parser.parse_args()

# print(args)

# if not args.pt:
#     pre_cmd = f'''
#     mkdir -p ~/datasets
#     mkdir -p ~/.hyperbox/nasbenchmark
#     ln -s /mount/cifar10 ~/datasets/
#     ln -s /mount/workspace/nasbench201.db ~/.hyperbox/nasbenchmark/
#     '''
#     os.system(pre_cmd)

options = vars(args)
for key in ['is_single_path', 'to_sample_similar', 'load_from_parent']:
    options[key] = [x!=0 for x in options[key]]
print(options)
keys = []
values = []
pass_keys = []
pass_values = []
for k,v in options.items():
    if not isinstance(v, list):
        pass_keys.append(k)
        pass_values.append(v)
    else:
        keys.append(k)
        values.append(v)
pass_options = dict(zip(pass_keys, pass_values))

opts = [dict(zip(keys,items)) for items in itertools.product(*values)]


try:
    num_gpus = torch.cuda.device_count()
    print(f"{num_gpus} gpus found")
except Exception as e:
    num_gpus = 1
    print(e)

if num_gpus <= 0:
    sys.exit("No gpus found")

num_subnets = 1
lr = 0.01
i = 0
num_cmds = 0
for opt in opts:
    opt.update(pass_options)

    split_criterion = opt['split_criterion']
    split_method = opt['split_method']
    split_num = opt['split_num']
    is_single_path = opt['is_single_path']
    to_sample_similar = opt['to_sample_similar']
    supernet_masks_path = opt['supernet_masks_path']
    similarity_method = opt['similarity_method']
    ID_method = opt['ID_method']
    load_from_parent = opt['load_from_parent']
    global_pool_path = opt['global_pool_path']
    warmup_epochs = opt['warmup_epochs']
    finetune_epoch = opt['finetune_epoch']
    network_cfg = opt['network_cfg']
    datamodule = opt['datamodule']
    other_cmds = opt['other_cmds']
    datamodule_map = {'cifar10_datamodule': 'c10', 'cifar100_datamodule': 'c100'}

    gpu_id = i%num_gpus

    if len(eval(split_num)) == 1:
        base_val = eval(split_num)[0]
        num_splits = base_val**len(warmup_epochs.split(','))
    else:
        num_splits = np.prod(eval(split_num))
    is_sp = 'sp' if is_single_path else 'fp'
    is_loadparent = 'loadparent' if load_from_parent else 'notloadparent'
    if load_from_parent:
        finetune_epoch //= 2
    dataset = datamodule_map[datamodule]
    if global_pool_path:
        suffix = f"{network_cfg}_{dataset}_global_cluster_sgdlr{lr}"
    elif supernet_masks_path is None:
        supernet_masks_path = 'null'
        suffix = f"{network_cfg}_{dataset}_{split_criterion}_{split_method}_{is_sp}_{num_splits}splits_{is_loadparent}_{num_subnets}nets_sgdlr{lr}"
    else:
        suffix = 'finetune_' + supernet_masks_path.split('/runs/')[-1].split('/')[0]
    if to_sample_similar:
        suffix += '_samplesimilar'

    others = "ipdb_debug=False logger.wandb.offline=True trainer.strategy=null trainer.limit_val_batches=0"
    if not is_single_path:
        repeat_num = 50
    else:
        repeat_num = 150
    if split_criterion == 'grad':
        repeat_num = 100

    if args.debug:
        finetune_epoch = 1
        warmup_epochs = "[1,2]"
        repeat_num = 1
        others += " trainer.fast_dev_run=True"
        suffix = 'debug_' + suffix

    others += f" engine.repeat_num={repeat_num}"
    others += f' engine.split_criterion={split_criterion}'
    others += f' engine.split_method={split_method}'
    others += f' ++engine.split_num={split_num}'
    if global_pool_path is None:
        global_pool_path = 'null'
    others += f' ++engine.global_pool_path={global_pool_path}'
    others += f' engine.is_single_path={is_single_path}'
    others += f' engine.warmup_epochs={warmup_epochs}' 
    others += f' engine.finetune_epoch={finetune_epoch}'
    others += f' ++engine.similarity_method={similarity_method}'
    others += f' ++engine.ID_method={ID_method}'
    others += f' ++engine.load_from_parent={load_from_parent}'
    others += f' ++engine.supernet_masks_path={supernet_masks_path}'
    others += f" ++model.num_subnets={num_subnets}"
    others += f" ++model.optimizer_cfg.lr={lr}"
    others += f" model/network_cfg={network_cfg}"
    others += f" datamodule={datamodule}"
    others += f" seed={i}"
    others += f" ++model.mutator_cfg.to_sample_similar={to_sample_similar}"
    if other_cmds is not None:
        others += f" {other_cmds}"
    cmd = f'''bash ./scripts/fewshot/fewshot_search_nb201.sh [{gpu_id}] {suffix} "{others}"  & sleep 10'''
    
    if i == len(opts)-1:
        cmd = cmd.replace('&', '')
    i += 1
    if not args.pt:
        os.system(cmd)
    print(cmd)
    num_cmds += 1

if not args.pt:
    os.system(cmd)
assert num_cmds == len(opts), f"num_cmds ({num_cmds}) != #opts ({len(opts)})"
print(f"{num_gpus} gpus found")
print(f"{num_cmds} commands to run")
print(options)
