from glob import glob
import os
import sys
import torch
import itertools

from argparse import ArgumentParser

# pre_cmd = f'''
# mkdir -p ~/datasets
# mkdir -p ~/.hyperbox/nasbenchmark
# ln -s /mount/cifar10 ~/datasets/
# ln -s /mount/workspace/nasbench201.db ~/.hyperbox/nasbenchmark/
# '''
# os.system(pre_cmd)


parser = ArgumentParser()
parser.add_argument("--split_criterion", nargs="+", type=str, default=["ID", "ID"])
parser.add_argument("--split_method", nargs="+", type=str, default=["mincut"])
parser.add_argument("--is_single_path", nargs="+", type=bool, default=[True, False])
parser.add_argument("--to_sample_similar", nargs="+", type=bool, default=[True])
parser.add_argument("--ID_method", type=str, default='lid')
parser.add_argument("--load_from_parent", nargs="+", type=bool, default=[True, False])
parser.add_argument("--warmup_epochs", nargs="+", type=str, default=["[50,75,90,100]"])
args = parser.parse_args()

print(args)

options = vars(args)
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

num_subnets = 1
lr = 0.001
i = 0
num_cmds = 0
for opt in opts:
    opt.update(pass_options)

    split_criterion = opt['split_criterion']
    split_method = opt['split_method']
    is_single_path = opt['is_single_path']
    to_sample_similar = opt['to_sample_similar']
    ID_method = opt['ID_method']
    load_from_parent = opt['load_from_parent']
    warmup_epochs = opt['warmup_epochs']
    finetune_epoch = 100

    gpu_id = i%num_gpus
    num_splits = 2**len(warmup_epochs.split(','))
    is_sp = 'sp' if is_single_path else 'fp'
    is_loadparent = 'loadparent' if load_from_parent else 'notloadparent'
    finetune_epoch = 20 if load_from_parent else 50
    suffix = f"nb201_c10_{split_criterion}_{split_method}_{is_sp}_{num_splits}splits_{is_loadparent}_{num_subnets}nets_adamlr{lr}"
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
    others += f" ++model.num_subnets={num_subnets}"
    others += f" ++model.optimizer_cfg.lr={lr}"
    others += f" ++model.mutator_cfg.to_sample_similar={to_sample_similar}"
    if split_criterion == 'ID':
        if not is_single_path:
            others += f" engine.repeat_num=20"
        else:
            others += f" engine.repeat_num=40"
    cmd = f'''bash ./scripts/fewshot/fewshot_search_nb201.sh {gpu_id} {suffix} "{others}"  & sleep 10'''
    
    if (i % 8 == 0 and i>0):
        cmd = cmd.replace('&', '')
    i += 1
    # os.system(cmd)
    print(cmd)
    num_cmds += 1
# os.system(cmd)
print(f"{num_gpus} gpus found")
print(f"{num_cmds} commands to run")
