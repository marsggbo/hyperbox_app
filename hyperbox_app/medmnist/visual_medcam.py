import argparse
import os
import cv2
import hydra
import kornia
import nibabel as nib
import numpy as np
import torch
from hyperbox.networks.utils import extract_net_from_ckpt
from omegaconf import OmegaConf
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.transform import resize
from torchvision import models


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument(
        '--cfg_path',
        type=str,
        help='config path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    from medcam import medcam
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = get_args()
    cfg_path = args.cfg_path
    cfg = OmegaConf.load(cfg_path)
    ncfg = cfg.model.network_cfg
    dcfg = cfg.datamodule
    size = 360
    ratio = 0.7
    csize = int(size*ratio)
    dcfg.img_size=[size,size]
    dcfg.center_size=[csize,csize]
    dcfg.batch_size=1
    dcfg.slice_num=32
    dcfg.num_workers=2

    model = hydra.utils.instantiate(ncfg)
    weight = extract_net_from_ckpt(
        '/home/comp/18481086/code/hyperbox_app/hyperbox_app/medmnist/reproduce/ccccii/auc98.57_acc93.86.ckpt'
    )
    model.load_state_dict(weight)
    model = model.to(device)
    datamodule = hydra.utils.instantiate(dcfg)

    layer = [
        # 'network.feature_mix_layer',
        'auto'
        # 'full'
    ]
    layer = 'auto'
    model = medcam.inject(
        model, output_dir="cam_ccccii", backend='gcam', layer=layer, label=1,
           save_maps=True, retain_graph=True, return_attention=True, enabled=True
    )
    model.eval()
    
    g2c = lambda gimg: kornia.color.grayscale_to_rgb(
        gimg.unsqueeze(0)).permute(1,2,0) # h*w*3
    dataloader = datamodule._data_loader(datamodule.dataset_test, True)
    for scan_idx, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device) # 1*1*d*h*w
        labels = labels.to(device)
        output = model(imgs)
        cams = model.medcam_dict['current_attention_map'] # 1*1*d*h*w
        cams = resize(cams[0][0], imgs.shape[-3:]) # d*h*w
        cams = cams[None,...]

        cam_img = imgs.cpu().numpy() + cams
        affine = np.array([[-7.40000010e-01, -0.00000000e+00,  0.00000000e+00,
         1.89082993e+02],
       [ 0.00000000e+00,  7.40000010e-01,  0.00000000e+00,
        -2.26166321e+02],
       [ 0.00000000e+00,  0.00000000e+00,  8.00000000e+00,
        -8.99200012e+02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
        ni_cam_img = nib.Nifti1Image(cam_img, affine=affine)
        nib.save(ni_cam_img, 'test.nii.gz')

        imgs = imgs.cpu()
        if imgs.shape[1] == 1: # channel is 1
            depth = imgs.shape[2]
            imgs_new = []
            for gimg in imgs[0][0]:
                gimg = g2c(gimg).numpy() # h*w*3
                imgs_new.append(gimg[None, ...])
            imgs = np.vstack(imgs_new) # d*h*w*3
        assert len(imgs.shape)==4 and imgs.shape[-1]==3

        dataset = 'ccccii'
        label = labels.view(-1).item()
        path = f"./cam_{dataset}/{label}/{scan_idx}"
        os.system(f'mkdir -p {path}')
        for slice_idx, (img, cam) in enumerate(zip(imgs, cams[0])):
            cam_image = show_cam_on_image(img, cam, use_rgb=True) # d*h*w
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            name = f"{path}/slice{slice_idx}.jpg"
            cv2.imwrite(name, cam_image)

        if scan_idx == 10:
            break