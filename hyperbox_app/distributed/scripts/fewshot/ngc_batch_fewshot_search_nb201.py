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
split_criterions = ['grad', 'ID']
# warmup_epochs_list = [
#     "[20,40]",
#     "[20,40,60,80]"
# ]
warmup_epochs_list = [
    "[50,100]",
    "[50,75,90,100]"
]
split_methods = ['mincut']
# split_methods = ['mincut', 'spectral_cluster']
is_single_paths = [True, False]
load_from_parent = False
# is_single_paths = [True]

try:
    num_gpus = torch.cuda.device_count()
    print(f"{num_gpus} gpus found")
except Exception as e:
    num_gpus = 1
    print(e)

num_subnets = 5
i = 0
num_cmds = 0
for split_criterion in split_criterions:
    for split_method in split_methods:
        for warmup_epochs in warmup_epochs_list:
            for is_single_path in is_single_paths:
                # is_single_path = True
                gpu_id = i%num_gpus
                num_splits = 2**len(warmup_epochs.split(','))
                is_sp = 'sp' if is_single_path else 'fp'
                is_loadparent = 'loadparent' if load_from_parent else 'notloadparent'
                suffix = f"nb201_c10_{split_criterion}_{split_method}_{is_sp}_{num_splits}splits_{is_loadparent}_{num_subnets}nets"
                others = "ipdb_debug=False logger.wandb.offline=True trainer.strategy=null trainer.limit_val_batches=0"
                if len(args) > 0:
                    others += f' {args[0]}'
                others += f' engine.split_criterion={split_criterion}'
                others += f' engine.split_method={split_method}'
                others += f' engine.is_single_path={is_single_path}'
                others += f' engine.warmup_epochs={warmup_epochs}' 
                others += f' ++engine.load_from_parent={load_from_parent}'
                others += f" model.num_subnets={num_subnets}"
                if split_criterion == 'ID':
                    if not is_single_path:
                        others += f" engine.repeat_num=5"
                    else:
                        others += f" engine.repeat_num=30"
                cmd = f'''bash ./scripts/fewshot/fewshot_search_nb201.sh {gpu_id} {suffix} "{others}"  & sleep 10'''
                
                if i == num_gpus-1 or (i+1) % num_gpus == 0:
                    cmd = cmd.replace('&', '')
                i += 1
                os.system(cmd)
                print(cmd)
                num_cmds += 1
print(f"{num_gpus} gpus found")
print(f"{num_cmds} commands to run")
