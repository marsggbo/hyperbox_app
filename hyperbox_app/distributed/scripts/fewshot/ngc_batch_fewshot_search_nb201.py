from glob import glob
import os
import sys

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
split_criterions = ['ID']
# split_criterions = ['grad', 'ID']
warmup_epochs_list = [
    "[20,40]",
    "[20,40,60,80]"
]
split_methods = ['mincut', 'spectral_cluster']
# is_single_paths = [True, False]
is_single_paths = [True]

i = 0
for split_criterion in split_criterions:
    for split_method in split_methods:
        # for is_single_path in is_single_paths:
        for warmup_epochs in warmup_epochs_list:
            is_single_path = True
            gpu_id = i%4
            num_splits = 2**len(warmup_epochs.split(','))
            if is_single_path:
                suffix = f"nb201_c10_{split_criterion}_{split_method}_sp_{num_splits}splits" 
            else:
                suffix = f"nb201_c10_{split_criterion}_{split_method}_fp_{num_splits}splits"
            others = "ipdb_debug=False logger.wandb.offline=True trainer.strategy=null trainer.limit_val_batches=0"
            if len(args) > 0:
                others += f' {args[0]}'
            others += f' engine.split_criterion={split_criterion}'
            others += f' engine.split_method={split_method}'
            others += f' engine.is_single_path={is_single_path}'
            others += f' engine.warmup_epochs={warmup_epochs}'
            if split_criterion == 'ID':
                if not is_single_path:
                    others += f" engine.repeat_num=5"
                else:
                    others += f" engine.repeat_num=50"
            cmd = f'''bash ./scripts/fewshot/fewshot_search_nb201.sh {gpu_id} {suffix} "{others}"  & sleep 10'''
            i += 1
            # if i % 4 == 0 and i > 0:
            if i == 3:
                cmd = cmd.replace('&', '')
            # os.system(cmd)
            print(cmd)