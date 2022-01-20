import os
import sys

import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize, rotate
import hydra
from omegaconf import OmegaConf
from hyperbox.networks.utils import extract_net_from_ckpt

class BaseCAM3D:
    def __init__(self, cfg, model):
        '''
        model: 1) glob_avgpool 2) fc
        '''
        self.cfg = cfg
        self.model = self.restore_model(model)
        self.parse_cfg()

    def parse_cfg():
        raise NotImplementedError

    def restore_model(self, model):
        try:
            model_path = self.cfg.cam.model_path
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f'loading from {model_path}')
        except:
            print(f'loading from none')
            pass
        return model

    def data_preprocess(self):
        raise NotImplementedError

    def register_hook(self, model, module_name):
        raise NotImplementedError

    def get_cam(self):
        '''the algorithm of generating CAM heat map
        return cam_map # cam_map.shape = scan.shape
        '''
        raise NotImplementedError

    def run(self):
        '''
        # data preprocessing
        scan = self.data_preprocess()
        bs, channel, depth, height, width = scan.shape
        # model
        model = self.register_hook(self.model, self.featmaps_module_name)
        # get CAM heat map
        cam_map = self.get_cam(model, preds)
        # visualize
        self.visualize(scan, cam, slice_id=-1, save_path='./')
        '''
        raise NotImplementedError

    @classmethod
    def visualize_slice(cls, slice_img, slice_cam, slicesssssssss='./slice.png'):
        save_path = os.path.dirname(filename)
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.axis('off')
        # mixed_img = slice_img + 0.2*slice_cam
        # mixed_img = mixed_img.astype(int)
        # plt.imshow(mixed_img)
        plt.imshow(slice_img, cmap=plt.cm.Greys_r, interpolation=None,
                vmax=1., vmin=0.)
        # slice_cam[slice_cam>0.3]=1
        slice_cam=(slice_cam-slice_cam.min())/(slice_cam.max()-slice_cam.min())
        # slice_cam=1-slice_cam
        slice_img=(slice_img-slice_img.min())/(slice_img.max()-slice_img.min())
        plt.imshow(slice_cam,
                interpolation=None, vmax=1., vmin=.0, alpha=0.3,
                cmap=plt.cm.gist_heat)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=10)
        plt.savefig(filename)
    
    @classmethod
    def visualize(cls, scan, cam, slice_id=-1, save_path='./cam'):
        if not os.path.exists(save_path): os.makedirs(save_path)
        depth, height, width = scan.shape
        if slice_id == -1:
            slice_indices = list(range(depth))
        else:
            slice_indices = [slice_id]

        for slice_id in slice_indices:
            slice_img = scan[slice_id, :, :].reshape(height, width)
            slice_cam = cam[slice_id, :, :].reshape(height, width)
            filename = os.path.join(save_path, f'slice{slice_id}.png')
            cls.visualize_slice(slice_img, slice_cam, filename)
                

class CAM3D(BaseCAM3D):
    def __init__(self, cfg_path, weight):
        '''
        model: 1) glob_avgpool 2) fc
        '''
        cfg = OmegaConf.load(cfg_path)
        ncfg = cfg.model.network_cfg
        dcfg = cfg.datamodule

        model = hydra.utils.instantiate(ncfg)
        weight = extract_net_from_ckpt(weight)
        model.load_state_dict(weight)
        self.model = model
        self.datamodule = hydra.utils.instantiate(dcfg)
        self.featmaps_module_name = 'network.global_avg_pooling' # 'network.feature_mix_layer'
        self.weights_module_name = 'network.classifier'
        self.save_path = f'./cam_{self.datamodule.__class__.__name__}'

    def register_hook(self, model, featmaps_module_name=None, weights_module_name=None):
        self.activation = {}

        # the weights of feat map
        if weights_module_name is None:
            weights_module_name = 'fc'
        try:
            self.activation['weights'] = list(model.network.modules())[-1].weight
        except Exception as e:
            print(str(e))
            raise NotImplementedError('last linear layer not found.')

        # feat maps
        def hook(model, input, output):
            self.activation['maps'] = input
        if featmaps_module_name is None:
            featmaps_module_name = 'glob_avgpool'
        try:
            self.handle = eval(f"model.{featmaps_module_name}.register_forward_hook(hook)")
        except:
            raise NotImplementedError('global avgpool not found.')

        return model

    def get_cam(self, scan, model, label=None):
        bs, channel, depth, height, width = scan.shape
        preds = model(scan)

        ## weights
        # params = list(model.parameters())[-2] # params of last linear layer
        # weights = np.squeeze(params.data.cpu().detach().numpy()) # class*c
        weights = self.activation['weights'].data.cpu().detach().numpy()

        # feat maps
        feat_maps = self.activation['maps'][0].cpu().detach().numpy() # c*d*h*w

        # cam
        feat_num = weights.shape[1]
        feat_size = feat_maps.shape[2:]
        # idx = int(self.label) if self.label>=0 else preds.argmax(1)
        # idx = preds.argmax(1)
        idx = label if label else preds.argmax(1)
        cam = np.dot(weights[idx].reshape(1, feat_num),
                            feat_maps[0].reshape(feat_num, -1)).reshape(1, *feat_size) # d*h*w
        cam = (cam - np.min(cam)) / np.max(cam)
        cam = resize(cam[0], (depth, width, height))
        return cam

    def run(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # data preprocessing
        # scan = self.data_preprocess(self.scan_path, self.transform, self.is_color, 
        #                             self.img_size, self.loader)
        dataloader = self.datamodule._data_loader(self.datamodule.dataset_test, True)
        for idx, (scans, labels) in enumerate(dataloader):
            scans = scans.to(device)
            break
        self.scans = scans.to(device)
        self.labels = labels

        # model
        model = self.register_hook(
            self.model, self.featmaps_module_name, self.weights_module_name)
        model = model.to(device)
        # model.eval()
        # model.train()

        for scan, label in zip(self.scans, self.labels):
            self.label = label
            scan = scan.unsqueeze(0)
            # cam
            cam = self.get_cam(scan, model, label)
            self.handle.remove()

            # visualize
            self.visualize(scan.cpu().numpy()[0,0,::], cam, -1, self.save_path)


if __name__ == '__main__':
    # [cfg_path, weight] = sys.argv[1:]
    cfg_path = '/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/reproduce/ccccii/.hydra/config.yaml'
    weight = '/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/reproduce/ccccii/auc98.57_acc93.86.ckpt'
    CAM = CAM3D(cfg_path, weight)
    CAM.run()