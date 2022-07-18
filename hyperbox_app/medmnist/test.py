import sys
from omegaconf import OmegaConf
from hydra.utils import instantiate

args = sys.argv[1:]

cfg_path = args[0]
mask_path = args[1]
cfg = OmegaConf.load(cfg_path)
net_cfg = cfg.model.network_cfg
net_cfg.mask = mask_path
net = instantiate(net_cfg)
print(net)