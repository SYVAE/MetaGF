#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil

from dataset.dataloader   import *
from config.args import arg_parser, arch_resume_names
from tools.op_counter import measure_model
from models.adaptive_inference import dynamic_evaluate
import models
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from tools.Train_utils import *
from models.SDN_Constructing import SDN
# from FixBN import fix_bn
args = arg_parser.parse_args()
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
import random
import numpy as np
from tqdm import tqdm
import cv2
from tools.DifUpdate import *

def main():

    global args
    print("2022.7.20  EMA-MetaGF...")
    print("MetaGF lr:{0}, Auxilr:{1} EMA_old:{2}".format(args.Metalr,args.auxiliarylr,args.EMAoldmomentum))
    time.sleep(5)
    baselineFile = args.comparedbaseline
    baseline=np.loadtxt(baselineFile,delimiter='\t',skiprows=1)
    baseline_acclist = []
    baseline_trainacclist = []
    for i in range(0, baseline.shape[0]):
        baseline_acclist.append(baseline[i, 5])
        baseline_trainacclist.append(baseline[i, 4])

    fig, ax = plt.subplots()
    ax.plot(np.array(range(0, len(baseline_acclist))), np.stack(baseline_acclist), '--', color='black', alpha=0.3)
    ax.plot(np.array(range(0, len(baseline_trainacclist))), np.stack(baseline_trainacclist), '-', color='black', alpha=0.3)

    ax.grid(axis='x', color='0.95')
    ax.legend(title='Parameter where:')
    ax.set_frame_on(False)
    ax.set_title('Acc')
    # plt.savefig("comparison.png")
    plt.pause(0.01)


    if not os.path.exists(args.save):
        os.makedirs(args.save)
    t = time.localtime()
    args.save = args.save + "/"+"Metalr{0}_Auxlr{1}_EMAold{2}".format(args.Metalr,args.auxiliarylr,args.EMAoldmomentum)+ str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(
        t.tm_mday) + '-' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + str(t.tm_sec) + "/"
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    print("seed:{0}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    print("model_savepath:{0}".format(args.save))
    best_err1, best_epoch = 0., 0
    if args.data.startswith('cifar'):
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.usingsdn:
        model = SDN(args)
    else:
        model = getattr(models, args.arch)(args)
    ValidNodeslist = []
    for depth in range(0, args.nBlocks):
        if args.usingsdn:
            model = SDN(args)
        else:
            model = getattr(models, args.arch)(args)
        data = torch.autograd.Variable(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)).cuda()

        model = torch.nn.DataParallel(model).cuda()
        model.train()
        output, _ = model(data)
        if not isinstance(output, list):
            output = [output]

        from collections import OrderedDict
        gradlist = OrderedDict()
        l = output[depth].sum()
        l.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                gradlist[n] = torch.tensor(1.0,dtype=torch.float).cuda()
                gradlist[n].requires_grad=True

        for name in model.state_dict():
            if not name in gradlist:
                featurename = name.split('.')[-1]
                if featurename in ["running_mean","running_var","num_batches_tracked"]:
                    gradlist[name] = torch.tensor(1.0, dtype=torch.float).cuda()
                    gradlist[name].requires_grad = False

        ValidNodeslist.append(gradlist)
        print(len(gradlist))
    for name in ValidNodeslist[0]:
        print(name)

    optimize_params = []
    initial=args.Metalr
    for depth in range(0, args.nBlocks):
        for i in ValidNodeslist[depth]:
            optimize_params.append({'params': ValidNodeslist[depth][i], 'initial_lr': initial})
    stepsizeOptimizer = torch.optim.Adam(optimize_params, lr=initial)
    stepSheduler = torch.optim.lr_scheduler.MultiStepLR(stepsizeOptimizer, [int(0.5 * args.epochs)], gamma=0.1,
                                                        last_epoch=args.epochs)
    if args.usingsdn:
        model = SDN(args)
    else:
        model = getattr(models, args.arch)(args)
    # print(model)
    print(IMAGE_SIZE)
    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    # print("------------------------------")
    print(n_flops, n_params)
    # print("------------------------------")
    torch.save(n_flops, os.path.join(args.save, 'flop.pth'))
    del(model)
    torch.save(args, os.path.join(args.save, 'args.pth'))

    # return

    if args.usingsdn:
        model = SDN(args)
    else:
        model = getattr(models, args.arch)(args)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    cudnn.benchmark = True
    # train_loader, val_loader, test_loader = get_dataloaders_minidatasets(args,200)
    train_loader, val_loader, test_loader = get_dataloaders(args)
    print("*************************************")
    print(args.use_valid, len(train_loader), len(val_loader), len(test_loader))
    print("*************************************")

    if args.evalmode is not None:
        m = torch.load(args.evaluate_from)
        model.load_state_dict(m['state_dict'])

        if args.evalmode == 'anytime':
            validate(test_loader, model, criterion)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return

    # set up logging
    global log_print, f_log
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)
    log_print('args:')
    log_print(args)
    print('model:', file=f_log)
    print(model, file=f_log)
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))

    f_log.flush()

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_err1'
              '\tval_err1\ttrain_err5\tval_err5']
    acclist = []
    trainacclist = []
    meta_optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    for epoch in range(args.start_epoch, args.epochs):
        print(">>>>reptile training1111")
        weightlist=[]
        for i in range(0,args.nBlocks):
            weightlist.append([])

        oldmodel=deepcopy(model)
        tasklist = list(range(0,args.nBlocks))
        print(tasklist)

        for taskid in tqdm(tasklist):
            lr = adjust_learning_rate(meta_optimizer, epoch, args, batch=0,
                                      nBatch=len(train_loader), method=args.lr_type)
            train_minibatch(train_loader, model, criterion, meta_optimizer, taskid,epoch)
            '''The ith task updatation'''
            weightlist[taskid]=(deepcopy(model.state_dict()))
            deepcopy_parameter(model,oldmodel)


        oldweightmodel=deepcopy(ValidNodeslist)

        adaptionmodel = deepcopy(model)
        innner_loop_state=deepcopy(model.state_dict())
        tmp=eval_validate(train_loader,adaptionmodel,criterion,ValidNodeslist,stepsizeOptimizer,weightlist,innner_loop_state,epoch)
        innner_loop_state=deepcopy(tmp)
        del tmp
        stepSheduler.step()

        with torch.no_grad():
            for depth in range(0, args.nBlocks):
                for i in ValidNodeslist[depth]:
                    ValidNodeslist[depth][i]=deepcopy((1-args.EMAoldmomentum)*ValidNodeslist[depth][i]+args.EMAoldmomentum*oldweightmodel[depth][i])



        model.load_state_dict(innner_loop_state)
        del adaptionmodel
        print(">>>>>meta updating")

        with torch.no_grad():
            train_loss, train_err1, train_err5, _ = eval(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        # val_loss, val_err1, val_err5 = validate(val_loader, model, criterion)
            val_loss, val_err1, val_err5 = validate(val_loader, model, criterion)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_err1, val_err1, train_err5, val_err5))
        trainacclist.append(train_err1)
        acclist.append(val_err1)

        # ax.step(np.array(range(0, len(acclist))), np.stack(acclist), where='post', label='post')
        ax.plot(np.array(range(0, len(acclist))), np.stack(acclist), '--', color='red', alpha=0.3)

        # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post',
        #         label='post')
        ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), '-', color='red',
                alpha=0.3)

        plt.pause(0.01)
        plt.savefig(args.save+"/acc")

        is_best = val_err1 > best_err1
        '''modified by sy::If you only plan to keep the best performing model (according to the acquired validation loss), 
        don’t forget that best_model_state = model.state_dict() returns a reference to the state and not its copy! You must 
        serialize best_model_state or use best_model_state = deepcopy(model.state_dict()) otherwise your best best_model_state 
        will keep getting updated by the subsequent training iterations. As a result, the final model state will be the 
        state of the overfitted model. '''

        saved_state = deepcopy(model.state_dict())
        if is_best:
            best_err1 = val_err1
            best_epoch = epoch
            print('Best var_err1 {}'.format(best_err1))

            model_filename = 'best_model.pth.tar'
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': saved_state,
                'best_err1': best_err1,
                'optimizer': optimizer.state_dict(),
                'routing_index': ValidNodeslist,
            }, args, is_best, model_filename, scores)

    model_filename = 'final.pth.tar'
    saved_state = deepcopy(model.state_dict())
    save_checkpoint({
        'epoch': args.epochs,
        'arch': args.arch,
        'state_dict': saved_state,
        'best_err1': best_err1,
        'optimizer': optimizer.state_dict(),
        'routing_index': ValidNodeslist,
    }, args, 0, model_filename, scores)
    print('Best val_err1: {:.4f} at epoch {}'.format(best_err1, best_epoch))

def deepcopy_parameter(model, oldmodel):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.data=deepcopy(oldmodel.state_dict()[n])


def eval(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.eval()
    end = time.time()

    running_lr = None
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):

            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output, _ = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            losslist=[]
            for j in range(len(output)):
                loss += criterion(output[j], target_var)
                # losslist.append(criterion(output[j], target_var))
            # measure error and record loss
            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                err1, err5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(err1.item(), input.size(0))
                top5[j].update(err5.item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Err@1 {top1.val:.4f}\t'
                      'Err@5 {top5.val:.4f}'.format(
                        epoch, i + 1, len(train_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr



def adapt_model(adaptionmodel,weightlist,innner_loop_state,ValidNodeslist):
    '''metaupdating now'''
    from collections import OrderedDict
    average_dict = OrderedDict()
    for name in innner_loop_state:
        add = 0
        weight_sum = 0
        for idx in range(0, len(weightlist)):
            # featurename = name.split('.')[-1]
            # if featurename in ["running_mean", "running_var", "num_batches_tracked"]:
            #     if torch.sum(weightlist[idx][name] - innner_loop_state[name]) == 0:
            #         continue
            if not ValidNodeslist[idx].__contains__(name):
                continue
            add += 1
            if average_dict.__contains__(name):
                # if featurename not in ["running_mean", "running_var", "num_batches_tracked"]:
                    average_dict[name] =average_dict[name]+ torch.sigmoid(ValidNodeslist[idx][name]) * weightlist[idx][name]
                    # assert (ValidNodeslist[idx][name] == 1)
                    weight_sum = weight_sum+torch.sigmoid(ValidNodeslist[idx][name])
                # else:
                #     average_dict[name] =average_dict[name]+ weightlist[idx][name]
                #     # assert (ValidNodeslist[idx][name] == 1)
                #     weight_sum =weight_sum+ torch.tensor(1.0, dtype=torch.float).cuda()
            else:
                # if featurename not in ["running_mean", "running_var", "num_batches_tracked"]:
                    average_dict[name] = torch.sigmoid(ValidNodeslist[idx][name])* weightlist[idx][name]
                    # assert (ValidNodeslist[idx][name] == 1)
                    weight_sum = torch.sigmoid(ValidNodeslist[idx][name])
                # else:
                #     average_dict[name] = weightlist[idx][name]
                #     # assert (ValidNodeslist[idx][name] == 1)
                #     weight_sum = torch.tensor(1.0, dtype=torch.float).cuda()
        '''calculating average'''
        # print(weight_sum)
        if average_dict.__contains__(name):
            average_dict[name] = average_dict[name] / weight_sum
        else:
            average_dict[name]=innner_loop_state[name]

    for name, p in adaptionmodel.named_parameters():
        p.update = (average_dict[name] - innner_loop_state[name])
    update_module(adaptionmodel)



def eval_validate(train_loader, model, criterion,ValidNodeslist,stepsizeOptimizer,weightlist,innner_loop_state,epoch):
    # switch to train mode
    loss=AverageMeter()
    stepsizeOptimizer.zero_grad()

    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    final_state=None
    for i, (input, target) in enumerate(train_loader):
        tmpmodel=deepcopy(model)
        # tmpmodel.eval()
        adapt_model(tmpmodel,weightlist,innner_loop_state,ValidNodeslist)

        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        output, _ = tmpmodel(input_var)
        if not isinstance(output, list):
            output = [output]

        max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])  # max tau_i --- C_i values
        cur_coeffs = 0.01 + epoch * (max_coeffs / args.epochs)  # to calculate the tau at the currect epoch
        cur_coeffs = np.minimum(max_coeffs, cur_coeffs)

        losslist = []
        tmploss = 0.0
        for idx in range(len(output)):
            if idx <= len(output) - 2:
                tmploss=tmploss+ float(cur_coeffs[idx]) *criterion(output[idx], target_var)
            else:
                tmploss = tmploss + criterion(output[idx], target_var)
            losslist.append(criterion(output[idx], target_var))
        # print("--here---")
        # tmploss =tmploss+ torch.nn.functional.max_pool1d(torch.stack(losslist).view(1, 1, -1), kernel_size=len(losslist)).sum()
        # stepsizeOptimizer.zero_grad()
        tmploss.backward()
        if i%1==0:
            stepsizeOptimizer.step()
            stepsizeOptimizer.zero_grad()
        # del tmpmodel
        '''UPDATING THE BN norm'''
        from collections import OrderedDict
        average_dict = OrderedDict()
        for name in innner_loop_state:
            featurename = name.split('.')[-1]
            if featurename in ["running_mean", "running_var", "num_batches_tracked"]:
                average_dict[name] = deepcopy(tmpmodel.state_dict()[name])
            else:
                average_dict[name] = deepcopy(innner_loop_state[name])

        model.load_state_dict(average_dict)
        innner_loop_state = deepcopy(average_dict)
        final_state=deepcopy(tmpmodel.state_dict())
        loss.update(tmploss.detach())

        losses.update(tmploss.detach().cpu())
        for j in range(len(output)):
            err1, err5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(err1.item(), input.size(0))
            top5[j].update(err5.item(), input.size(0))
        if i % 10 == 0:
            print('Epoch(adapt):{0}',
                  'Loss {loss.val:.4f}\t'
                  'Err@1 {top1.val:.4f}\t'
                  'Err@5 {top5.val:.4f}'.format(epoch, loss=losses, top1=top1[-1], top5=top5[-1]))
    stepsizeOptimizer.step()

    return final_state

def train_minibatch(train_loader, model, criterion, optimizer, task,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, _ = model(input_var)
        if not isinstance(output, list):
            output = [output]


        if not args.usingsdn:
            loss = criterion(output[task], target_var)
            tmpweight = 1.0
        else:
            max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])  # max tau_i --- C_i values
            cur_coeffs = 0.01 + epoch * (max_coeffs / args.epochs)  # to calculate the tau at the currect epoch
            cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
            if task <= len(output) - 2:
                loss = float(cur_coeffs[task]) * criterion(output[task], target_var)
                tmpweight = float(cur_coeffs[task])
            else:
                loss = criterion(output[task], target_var)
                tmpweight = 1.0

        '''auxiliary training'''
        for j in range(len(output)):
            weight = args.auxiliarylr * np.array([1, 1, 0.6, 0.6, 0.75, 0.9, 1])
            if j != task:
                tmp = weight[j] * tmpweight * criterion(output[j], target_var)
                loss += tmp


        losses.update(loss.item(), input.size(0))
        for j in range(len(output)):
            err1, err5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(err1.item(), input.size(0))
            top5[j].update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 and task==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Err@1 {top1.val:.4f}\t'
                  'Err@5 {top5.val:.4f}'.format(
                epoch, i + 1, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                loss=losses, top1=top1[task], top5=top5[task]))


    return 0


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            # compute output
            output,_ = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)
            # measure error and record loss
            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                err1, err5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(err1.item(), input.size(0))
                top5[j].update(err5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Err@1 {top1.val:.4f}\t'
                      'Err@5 {top5.val:.4f}'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1]))
                # break
    for j in range(args.nBlocks):
        print(' * Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
        """
        print('Exit {}\t'
              'Err@1 {:.4f}\t'
              'Err@5 {:.4f}'.format(
              j, top1[j].avg, top5[j].avg))
        """
    # print(' * Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg

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

if __name__ == '__main__':
    # torch.cuda.set_device(2)
    main()
