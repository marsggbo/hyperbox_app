import argparse
import os
import random
import shutil
import time
import warnings
import importlib
from enum import Enum
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms as T
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset

from loguru import logger
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hyperbox_app.distributed import networks
from hyperbox.mutator import RandomMutator
from hyperbox.utils.utils import load_json, save_arch_to_json
from hyperbox.utils.logger import get_logger


model_names = ['proxylessnas', 'nb201', 'spos', 'ofa']
# model_names += sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

hx_net_path = os.path.dirname(networks.__file__).replace('networks', 'configs')

arch2cfg = lambda name: os.path.join(hx_net_path, f'model/network_cfg/{name}.yaml')
arch_maps = {name: arch2cfg(name) for name in model_names}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/xihe/datasets/imagenet2012',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--ofa_dek', type=str, default='4,6,7', help='OFA depth, expansion and kernel size')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

def main():
    args = parser.parse_args()
    log_dir = f'./logs/runs/finetune_{args.arch}_{args.suffix}/{time.strftime("%Y%m%d-%H%M%S")}'
    args.log_dir = log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.system(f'cp {args.mask} {log_dir}')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    model_cfg = arch_maps[args.arch]
    model_cfg = OmegaConf.load(model_cfg)
    model_cfg.num_classes = 1000
    if args.arch == 'ofa':
        model_cfg.pretrained_weights = '/home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/networks/ofa/hyperbox_OFA_MBV3_k357_d234_e46_w1.pth'
        supernet = instantiate(model_cfg)
        depth, expand_ratio, kernel_size = [int(x) for x in args.ofa_dek.split(',')]
        mask = supernet.gen_mask(depth=depth, expand_ratio=expand_ratio, kernel_size=kernel_size)
        # mask = supernet.gen_mask(depth=4, expand_ratio=6, kernel_size=7)
        tmp_mask_path = os.path.join(args.log_dir, 'mask.json')
        save_arch_to_json(mask, tmp_mask_path)
        subnet_cfg = deepcopy(model_cfg)
        subnet_cfg.pretrained_weights = None
        subnet_cfg.mask = tmp_mask_path
        model = instantiate(subnet_cfg)
        model.load_from_supernet(supernet.state_dict())
    else:
        model_cfg.mask = args.mask
        model = instantiate(model_cfg)
    # mask = load_json(args.mask)
    # mutator = RandomMutator(model)
    # mutator.sample_by_mask(mask)
    
    log = get_logger(os.path.join(args.log_dir, f'log{args.rank}.txt'), is_rank_zero=False)

    if not torch.cuda.is_available():
        log.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to('cpu')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.finetune:
        try:
            for name, p in model.named_parameters():
                if 'classifier' in name:
                    p.requires_grad = True
                    log.info(f'freezing the weights of {name}')
                else:
                    p.requires_grad = False
                    
        except Exception as e:
            log.info(e)
            log.info('freeze params failed')
            

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    log.info('=> loading data files')
    train_ops = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
    ]
    if args.arch == 'ofa':
        policy = T.AutoAugmentPolicy.IMAGENET
        augmenter = T.AutoAugment(policy)
        train_ops.append(augmenter)
        log.info('=> using AutoAugment')
    else:
        train_ops.append(ColorJitter(brightness=32. / 255., saturation=0.5))
        log.info('=> using ColorJitter')
    train_ops.extend([
        transforms.ToTensor(),
        normalize
    ])
        
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(train_ops)
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        log.info('start evaluating')
        validate(val_loader, model, criterion, args)
        return
    if args.debug:
        args.epochs = 2
    log.info('start training')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        
        scheduler.step()

        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            filename = os.path.join(args.log_dir, 'checkpoint.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, filename)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f', args=args)
    data_time = AverageMeter('Data', ':6.3f', args=args)
    losses = AverageMeter('Loss', ':.4e', args=args)
    top1 = AverageMeter('Acc@1', ':6.2f', args=args)
    top5 = AverageMeter('Acc@5', ':6.2f', args=args)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch), args=args)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.debug and i == 10:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                if args.debug and i == 10:
                    break
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE, args=args)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE, args=args)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE, args=args)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE, args=args)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ', args=args)

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        base_dir = os.path.dirname(filename)
        best_filename = os.path.join(base_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE, args=None):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.args = args
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count]).to(self.args.rank)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", args=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log = get_logger(os.path.join(args.log_dir, f'log{args.rank}.txt'), is_rank_zero=False)

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.log.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        self.log.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
    
# python finetune_masks/finetune_imagenet.py \
# --dist-url 'tcp://127.0.0.1:8888' --dist-backend 'nccl' --multiprocessing-distributed \
# --world-size 1 --rank 0 --data /home/xihe/datasets/imagenet2012 --epochs 200 \
#  -a proxylessnas \
# --mask /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/finetune_masks/proxyless_masks/ID/v1/0_0.6813.json 

# python finetune_masks/finetune_imagenet.py \
# -a ofa --dist-url 'tcp://127.0.0.1:8888' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
# --epochs 200 --lr 0.0001 --suffix d4e6k7 --ofa_dek 4,6,7