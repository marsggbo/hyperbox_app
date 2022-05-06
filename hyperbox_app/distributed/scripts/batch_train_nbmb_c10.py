from glob import glob
import os
import random

template = '''
bash scripts/train_nbmb_c10.sh {} \
{} \
"{}" &
sleep 2
'''

i = 0
for sample_interval in [1]:
# for sample_interval in [1, 5, 10]:
    # for num_subnets in [1]:
    for num_subnets in [1, 3, 6, 12]:
        for set_to_none in [True, False]:
        # set_to_none = True
            gpu_id = i % 4
            suffix = f'{num_subnets}net_{sample_interval}batch'
            if set_to_none:
                suffix += '_NoneGrad'
            other = f"+model.num_subnets={num_subnets} +model.sample_interval={sample_interval} " \
                + f"+model.set_to_none={set_to_none} seed={i+random.randint(0,10000)} " \
                # + "ipdb_debug=True logger.wandb.offline=True hydra.job.name=debug "
            cmd = template.format(gpu_id, suffix, other)
            if i % 4 == 3:
                cmd = cmd.replace('&', '')
            print(cmd)
            os.system(cmd)
            i += 1
