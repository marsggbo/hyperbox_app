from glob import glob
import os
import random

#  gpu_id, suffix, others
template = '''
bash scripts/train_nb201_c10.sh {} {} "{}"  &
sleep 2" 
'''

BS = 16
i = 2
for sample_interval in [1]:
    for num_subnets in [1, 1]:
        bs = BS // num_subnets
        for set_to_none in [False]:
        # set_to_none = True
            gpu_id = i % 4
            suffix = f'c10_{num_subnets}net_{sample_interval}batch'
            if set_to_none:
                suffix += '_NoneGrad'
            other = f"+model.num_subnets={num_subnets} +model.sample_interval={sample_interval} " \
                + f"+model.set_to_none={set_to_none} seed={i+random.randint(0,10000)} trainer.max_epochs=3600" \
                # + "ipdb_debug=True logger.wandb.offline=True hydra.job.name=debug "
            cmd = template.format(gpu_id, suffix, other)
            if i % 4 == 3:
                cmd = cmd.replace('&', '')
            print(cmd)
            # os.system(cmd)
            i += 1

# ckpts = glob('/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/*batch*/*/checkpoints/last*')
# num = len(ckpts)
# i = 0
# for idx, ckpt in enumerate(ckpts[:1]):
#     # print(ckpt)
#     # /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201_gpunum1_c10_1net_1batch/2022-04-27_09-20-46/checkpoints/last.ckpt
#     gpu_id = i%4
#     suffix = ckpt.split('runs/')[1].split('/')[0]
#     valid_batches = 50
#     # suffix = suffix[0] + '_' + suffix[1]
#     cmd = template.format(gpu_id, suffix, valid_batches, ckpt)
#     i += 1
#     print(f"{cmd}\n{idx+1}/{num}")
#     os.system(cmd)
