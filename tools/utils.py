import sys
import time
import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch
from torch.autograd import Variable
from functools import reduce
import operator


count_ops = 0
count_params = 0

class KDLoss(nn.Module):
    def __init__(self, args):
        super(KDLoss, self).__init__()
        
        self.kld_loss = nn.KLDivLoss().cuda()
        self.ce_loss = nn.CrossEntropyLoss().cuda()
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        self.T = args.T
        self.gamma = args.gamma
        self.nBlocks = args.nBlocks

    def loss_fn_kd(self, outputs, targets, soft_targets):
        loss = self.ce_loss(outputs[-1], targets)
        T = self.T
        for i in range(self.nBlocks - 1):
            _ce = (1. - self.gamma) * self.ce_loss(outputs[i], targets)
            _kld = self.kld_loss(self.log_softmax(outputs[i] / T), self.softmax(soft_targets.detach() / T)) * self.gamma * T * T
            loss = loss + _ce + _kld
        return loss


class MyRandomSizedCrop(object):
    def __init__(self, size, augmentation=0.08, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.augmentation = augmentation

    def __call__(self, img):
        for _ in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.augmentation, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = transforms.Scale(self.size, interpolation=self.interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(scale(img))


def create_save_folder(save_path, ignore_patterns=[]):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('create folder: ' + save_path)



def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return

def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.lr,
                                beta=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res





def adjust_learning_rate_toy(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):

    lr, decay_rate = args.lr, 0.1
    if epoch >= args.epochs * 0.75:
        lr *= decay_rate ** 2
    elif epoch >= args.epochs * 0.5:
        lr *= decay_rate
    try:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    except:
        for param_group in optimizer.optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.contiguous()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res