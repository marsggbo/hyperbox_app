import argparse
import cv2
import numpy as np
import torch
import kornia
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import hydra
from omegaconf import OmegaConf
from hyperbox.networks.utils import extract_net_from_ckpt


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = get_args()
    cfg_path = args.cfg_path
    cfg = OmegaConf.load(cfg_path)
    ncfg = cfg.model.network_cfg
    dcfg = cfg.datamodule
    dcfg.img_size=[128,128]
    dcfg.center_size=[128,128]
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
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    model.eval()
    # model = models.resnet50(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [
        model.network.blocks[-3],
        model.network.blocks[-2],
        model.network.blocks[-1],
        model.network.feature_mix_layer
    ]

    dataloader = datamodule._data_loader(datamodule.dataset_test, True)
    for idx, (imgs, labels) in enumerate(dataloader):
        imgs = imgs
        labels = labels
        break

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(1)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    for scan_idx, (rgb_img, label) in enumerate(zip(imgs, labels)):
        # rbg_img: c*d*h*w
        input_tensor = rgb_img.unsqueeze(0).to(device) # bs*c*d*h*w (bs=1)
        rgb_img = rgb_img.detach().numpy()

        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth) # bs*d*h*w (bs=1)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :] # d*h*w

            g2c = lambda gimg: kornia.color.grayscale_to_rgb(
                torch.from_numpy(gimg).unsqueeze(0)).permute(1,2,0) # h*w*3
            if rgb_img.shape[0] == 1: # channel is 1
                depth = rgb_img.shape[1]
                rgb_img_new = []
                for gimg in rgb_img[0]:
                    gimg = g2c(gimg).numpy() # h*w*3
                    rgb_img_new.append(gimg[None, ...])
                rgb_img = np.vstack(rgb_img_new) # d*h*w*3
            assert len(rgb_img.shape)==4 and rgb_img.shape[-1]==3
    
            for slice_idx, (img, cam) in enumerate(zip(rgb_img, grayscale_cam)):
                cam_image = show_cam_on_image(img, cam, use_rgb=True) # d*h*w
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                dataset = 'ccccii'
                name = f'./cam_ccccii/{args.method}_{dataset}_label{label}_scan{scan_idx}_slice{slice_idx}.jpg'
                cv2.imwrite(name, cam_image)
