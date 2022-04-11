# hyperbox_app

# Dependency

```
pip install -r requirements.txt
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
