# hyperbox_app

# Dependency

- 1. install dependencies

```
pip install -r requirements.txt
```

- 2. install `hyperbox_app`

```
cd /path/to/hyperbox_app
python setup.py develop
```

# Run

```bash
python -m hyperbox.run \
hydra.searchpath=[/path/to/hyperbox_app/hyperbox_app/medmnist/configs] \
experiment=gdas_nii.yaml \
trainer.gpus=1 \
...
```

# References

- https://github.com/marsggbo/hyperbox/wiki/Hydra-Q&A
