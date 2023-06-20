# Installs

```
pip install -U catalyst
```

# Search & Train & Test
```
python -m hyperbox.run hydra.searchpath=[/home/xihe/xinhe/hyperbox_app/hyperbox_app/medmnist/configs] experiment=fracture3d_daresnet.yaml trainer.gpus=1 trainer.accelerator=gpu
```

# Ablation Study

## Search Data Augmentation (DA) and Neural Architecture (NA) Separately

```
python -m hyperbox.run hydra.searchpath=[/home/xihe/xinhe/hyperbox_app/hyperbox_app/medmnist/configs] experiment=fracture3d_daresnet.yaml model.network_cfg._target_=hyperbox_app.medmnist.networks.mbnet_fixedAug.DAMobile3DNetFixedAug +model.network_cfg.aug_mask=/path/to/arch.json
```

You only need to modify:
- experiment=fracture3d_daresnet.yaml
- +model.network_cfg.aug_mask=/path/to/arch.json