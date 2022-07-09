from glob import glob
import os
import sys
import torch

args = sys.argv[1:]
# if len(args) == 1:
#     path_pattern = args[0]
#     path_pattern += '*/*/check*'
# else:
#     path_pattern = '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/fewshot_search_nb201_*/*/check*'

# paths = glob(path_pattern)
# print(len(paths), 'paths to evaluate')

pre_cmd = f'''
mkdir -p ~/datasets
mkdir -p ~/.hyperbox/nasbenchmark
ln -s /mount/cifar10 ~/datasets/
ln -s /mount/workspace/nasbench201.db ~/.hyperbox/nasbenchmark/
'''
os.system(pre_cmd)

# split_criterions = ['grad']
# split_criterions = ['ID']
# split_criterions = ['grad', 'grad']
split_criterions = ['ID', 'ID']
# split_criterions = ['grad', 'ID']
# warmup_epochs_list = [
#     "[20,40]",
#     "[20,40,60,80]"
# ]
warmup_epochs_list = [
    # "[50,100]",
    "[50,75,90,100]"
]
finetune_epoch = 100
split_methods = ['mincut']
# split_methods = ['mincut', 'spectral_cluster']
# is_single_paths = [True, False]
if len(args) == 1:
    is_single_paths = [eval(x) for x in args[0].split(',')]
else:
    is_single_paths = [True, False]
# is_single_paths = [False, True]
load_from_parents = [False, True]
to_sample_similar = True
ID_method = 'lid'
# is_single_paths = [True]

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
for split_criterion in split_criterions:
    for split_method in split_methods:
        for warmup_epochs in warmup_epochs_list:
            for is_single_path in is_single_paths:
                for load_from_parent in load_from_parents:
                    gpu_id = i%num_gpus
                    num_splits = 2**len(warmup_epochs.split(','))
                    is_sp = 'sp' if is_single_path else 'fp'
                    is_loadparent = 'loadparent' if load_from_parent else 'notloadparent'
                    suffix = f"nb201_c10_{split_criterion}_{split_method}_{is_sp}_{num_splits}splits_{is_loadparent}_{num_subnets}nets_adamlr{lr}"
                    if to_sample_similar:
                        suffix += '_samplesimilar'
                    others = "ipdb_debug=False logger.wandb.offline=True trainer.strategy=null trainer.limit_val_batches=0"
                    # if len(args) > 0:
                    #     others += f' {args[0]}'
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
                            others += f" engine.repeat_num=5"
                        else:
                            others += f" engine.repeat_num=20"
                    cmd = f'''bash ./scripts/fewshot/fewshot_search_nb201.sh {gpu_id} {suffix} "{others}"  & sleep 10'''
                    
                    # if i == num_gpus-1 or (i % (num_gpus*2) == 0 and i>0):
                    #     cmd = cmd.replace('&', '')
                    i += 1
                    os.system(cmd)
                    print(cmd)
                    num_cmds += 1
os.system(cmd)
print(f"{num_gpus} gpus found")
print(f"{num_cmds} commands to run")
