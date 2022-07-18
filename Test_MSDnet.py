#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil
from dataset.dataloader import get_dataloaders
from config.args import arg_parser
from models.adaptive_inference import dynamic_evaluate
import models
from tools.op_counter import measure_model
from tools.utils import *

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

torch.manual_seed(args.seed)


def main():
    global args
    # args.save="/home/sunyi/sy/A6Gradient/IMTA/results/11/save_models/"
    args.evaluate_from=args.save+"save_models/best_model.pth.tar"
    print(args.evaluate_from)
    time.sleep(2)
    # args.evalmode='anytime'
    args.evalmode='dynamic'
    #args.data_root='/media/sunyi/E/Saliency/cifar/'

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224

    model = getattr(models, args.arch)(args)
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del (model)

    model = getattr(models, args.arch)(args)
    model.cuda()
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()



    cudnn.benchmark = True
    train_loader, val_loader, test_loader = get_dataloaders(args)
    try:
        state_dict = torch.load(args.evaluate_from,map_location="cuda:0")['state_dict']
    except:
        state_dict = torch.load(args.evaluate_from,map_location="cuda:0",encoding='ascii')
        print("here")
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss().cuda()
    if args.evalmode == 'anytime':
        print("anytime prediction")
        validate(test_loader, model, criterion)
    else:
        dynamic_evaluate(model, test_loader, val_loader, args)

    validate(test_loader, model, criterion)

    return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output,_ = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                    i + 1, len(val_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))

    filename=args.save+'/'+"anytime.txt"
    with open(filename,"w") as f:
        for j in range(args.nBlocks):
            print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
            f.write("{0} {1}\n".format(top1[j].avg, top5[j].avg))
        # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg





def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


if __name__ == '__main__':
    main()
