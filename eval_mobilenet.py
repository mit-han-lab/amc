# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import argparse

from torch.autograd import Variable

from models.mobilenet import MobileNet
from lib.utils import AverageMeter, progress_bar, accuracy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='mobilenet_0.5flops', type=str, help='name of the model to test')
parser.add_argument('--imagenet_path', default=None, type=str, help='Directory of ImageNet')
parser.add_argument('--n_gpu', default=1, type=int, help='name of the job')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()


def get_dataset():
    # lazy import
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    if not args.imagenet_path:
        raise Exception('Please provide valid ImageNet path!')
    print('=> Preparing data..')
    valdir = os.path.join(args.imagenet_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    input_size = 224
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_worker, pin_memory=True)
    n_class = 1000
    return val_loader, n_class


def get_model(n_class):
    print('=> Building model {}...'.format(args.model))
    if args.model == 'mobilenet_0.5flops':
        net = MobileNet(n_class, profile='0.5flops')
        checkpoint_path = './checkpoints/mobilenet_imagenet_0.5flops_70.5.pth.tar'
    else:
        raise NotImplementedError

    print('=> Loading checkpoints..')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])  # remove .module

    return net


def evaluate():
    # build dataset
    val_loader, n_class = get_dataset()
    # build model
    net = get_model(n_class)

    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net, list(range(args.n_gpu)))
        cudnn.benchmark = True

    # begin eval
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(batch_idx, len(val_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))


if __name__ == '__main__':
    evaluate()