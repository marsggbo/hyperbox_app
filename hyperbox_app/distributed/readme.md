
# Example 1

- NASBench201Network
- RandomMutator

## 1. Supernet Training

```
bash scripts/train_nb201_c10.sh 0 "c10_5net_1batch_watch_NoneGrad" "+model.num_subnets=5 +model.sample_interval=1 seed=1 +callbacks.watch_model._target_=hyperbox.callbacks.wandb_callbacks.WatchModel +model.set_to_none=True" 
```

## 2. EA search

将实验配置文件`config.yaml`拷贝到一个新的路径,假设是`/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/temp_space/nb201/cfg_c10_ea_new`，然后对该路径下`config.yaml`作如下修改：

- 修改`model.mutator_cfg`为`EvolutionMutator`
- 修改`engine`为`EASearchNB201`

```
model:
  mutator_cfg:
    _target_: hyperbox.mutator.evolution_mutator.EvolutionMutator
    warmup_epochs: 0
    evolution_epochs: 100
    population_num: 50
    selection_alg: 'nsga2'
    selection_num: 0.2
    crossover_num: 0.4
    crossover_prob: 0.1
    mutation_num: 0.4
    mutation_prob: 0.1
    flops_limit: 5000 # MFLOPs
    size_limit: 80 # MB
    log_dir: 'evolution_logs'
    topk: 10
    to_save_checkpoint: True
    to_plot_pareto: True
    figname: 'evolution_pareto.pdf'
engine:
  __target_: hyperbox_app.distributed.engine.ea_searchNB201.EASearchNB201
```

运行

```
bash scripts/ea_search_nb201.sh \
0 \
nb201_gpunum1_c10_1net_1batch \
10 \
/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/logs/runs/search_nb201_gpunum1_c10_1net_1batch/2022-04-27_09-20-46/checkpoints/last.ckpt \
"ipdb_debug=True logger.wandb.offline=True"
```

## 3. Finetuning

