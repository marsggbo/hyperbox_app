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


split_criterions = ['grad']
# split_criterions = ['ID']
# split_criterions = ['grad', 'ID']
split_methods = ['mincut', 'spectral_cluster']
is_single_paths = [True, False]

i = 0
for split_criterion in split_criterions:
    for split_method in split_methods:
        for is_single_path in is_single_paths:
            gpu_id = i%4
            if is_single_path:
                suffix = f"nb201_c10_{split_criterion}_{split_method}_sp_16splits" 
            else:
                suffix = f"nb201_c10_{split_criterion}_{split_method}_fp_16splits"
            others = "ipdb_debug=False logger.wandb.offline=True trainer.strategy=null trainer.limit_val_batches=0"
            if len(args) > 0:
                others += f' {args[0]}'
            others += f' engine.split_criterion={split_criterion}'
            others += f' engine.split_method={split_method}'
            others += f' engine.is_single_path={is_single_path}'
            if split_criterion == 'ID':
                if not is_single_path:
                    others += f" engine.repeat_num=5"
                else:
                    others += f" engine.repeat_num=50"
            cmd = f'''bash ./scripts/fewshot/fewshot_search_nb201.sh {gpu_id} {suffix} "{others}"  & sleep 10'''
            i += 1
            if i % 8 == 0 and i > 0:
                cmd = cmd.replace('&', '')
            print(f"{cmd}")
            # os.system(cmd)
print(f'{i} commands to run')