from glob import glob
import os
import sys
import torch
import itertools

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--split-method", nargs="+", type=str, default=["mincut"])
parser.add_argument("--similarity-method", nargs="+", type=str, default=["cosine"])
parser.add_argument("--split-criterion", nargs="+", type=str, default=["ID", "ID"])
parser.add_argument("--is-single-path", nargs="+", type=int, default=[1, 1])
parser.add_argument("--to-sample-similar", nargs="+", type=int, default=[1])
parser.add_argument("--load-from-parent", nargs="+", type=int, default=[1, 0])
parser.add_argument("--ID-method", type=str, default='lid')
parser.add_argument("--warmup-epochs", nargs="+", type=str, default=["[50,75,90,100]"])
parser.add_argument("--debug", action='store_true')
parser.add_argument("--supernet-masks-path", type=str, default=None)
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
    is_single_path = opt['is_single_path']
    to_sample_similar = opt['to_sample_similar']
    supernet_masks_path = opt['supernet_masks_path']
    ID_method = opt['ID_method']
    load_from_parent = opt['load_from_parent']
    warmup_epochs = opt['warmup_epochs']
    finetune_epoch = 100
    if args.debug:
        finetune_epoch = 1
        warmup_epochs = "[1,2]"

    gpu_id = i%num_gpus
    num_splits = 2**len(warmup_epochs.split(','))
    is_sp = 'sp' if is_single_path else 'fp'
    is_loadparent = 'loadparent' if load_from_parent else 'notloadparent'
    finetune_epoch = 50 if load_from_parent else 100
    if supernet_masks_path is None:
        supernet_masks_path = 'null'
        suffix = f"nb201_c10_{split_criterion}_{split_method}_{is_sp}_{num_splits}splits_{is_loadparent}_{num_subnets}nets_sgdlr{lr}"
    else:
        suffix = 'finetune_' + supernet_masks_path.split('/runs/')[-1].split('/')[0]
    if to_sample_similar:
        suffix += '_samplesimilar'

    others = "ipdb_debug=False logger.wandb.offline=True trainer.strategy=null trainer.limit_val_batches=0"
    others += f' engine.split_criterion={split_criterion}'
    others += f' engine.split_method={split_method}'
    others += f' engine.is_single_path={is_single_path}'
    others += f' engine.warmup_epochs={warmup_epochs}' 
    others += f' engine.finetune_epoch={finetune_epoch}'
    others += f' ++engine.ID_method={ID_method}'
    others += f' ++engine.load_from_parent={load_from_parent}'
    others += f' ++engine.supernet_masks_path={supernet_masks_path}'
    others += f" ++model.num_subnets={num_subnets}"
    others += f" ++model.optimizer_cfg.lr={lr}"
    others += f" ++model.mutator_cfg.to_sample_similar={to_sample_similar}"
    if split_criterion == 'ID':
        if not is_single_path:
            others += f" engine.repeat_num=20"
        else:
            others += f" engine.repeat_num=40"
    if args.debug:
        others += " trainer.fast_dev_run=True"
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
