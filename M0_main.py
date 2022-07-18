#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys

sys.path.insert(1, '../')
from dataset.dataloader import *
from config.args import arg_parser
from tools.op_counter import measure_model
import models
from tools.Train_utils import *
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from models.SDN_Constructing import SDN
import time
import random
from copy import deepcopy
args = arg_parser.parse_args()
# args.data='cifar10'
# args.nBlocks = 7
# args.stepmode = "even"
# args.step = 2
# args.base = 4
# args.grFactor = "1-2-4"
# args.bnFactor = "1-2-4"
# args.growthRate = 16
# args.nChannels = 16
# args.data_root= "../data/cifar/"
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
'''recording the importance of the shared parameter to different eixt'''
'''tracking the gradient norm'''


def main():
    global args
    print("baseline")
    time.sleep(5)

    fig, ax = plt.subplots()
    ax.grid(axis='x', color='0.95')
    ax.legend(title='Parameter where:')
    ax.set_frame_on(False)
    ax.set_title('Acc')
    plt.pause(0.01)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    t = time.localtime()

    args.save = args.save + "/" + str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(
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
    del (model)
    torch.save(args, os.path.join(args.save, 'args.pth'))

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

    cudnn.benchmark = True
    # train_loader, val_loader, test_loader = get_dataloaders_minidatasets(args, 200)
    train_loader, val_loader, test_loader = get_dataloaders(args)
    print("*************************************")
    print(args.use_valid, len(train_loader), len(val_loader), len(test_loader))
    print("*************************************")


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
    conflict_matrixlist = []
    Conflictlist = []


    for epoch in range(args.start_epoch, args.epochs):
        weightlist = []
        '''Train the model'''
        train_loss, train_err1, train_err5, lr = train(train_loader, model, criterion, optimizer, epoch)
        with torch.no_grad():
            train_loss, train_err1, train_err5, _ = eval(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_err1, val_err5 = validate(val_loader, model, criterion)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_err1, val_err1, train_err5, val_err5))
        trainacclist.append(train_err1)
        acclist.append(val_err1)

        plt.figure(1)
        # ax.step(np.array(range(0, len(acclist))), np.stack(acclist), where='post', label='post')
        ax.plot(np.array(range(0, len(acclist))), np.stack(acclist), '--', color='red', alpha=0.3)

        # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post',
        #         label='post')
        ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), '-', color='red',
                alpha=0.3)

        plt.pause(0.01)
        plt.savefig(args.save + "/acc")

        is_best = val_err1 > best_err1
        '''modified by sy::If you only plan to keep the best performing model (according to the acquired validation loss), 
        don’t forget that best_model_state = model.state_dict() returns a reference to the state and not its copy! You must 
        serialize best_model_state or use best_model_state = deepcopy(model.state_dict()) otherwise your best best_model_state 
        will keep getting updated by the subsequent training iterations. As a result, the final model state will be the 
        state of the overfitted model. '''

        saved_state = deepcopy(model.state_dict())
        if is_best:
            saved_state = deepcopy(model.state_dict())

            optimizer_state = deepcopy(optimizer.state_dict())

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
                    'optimizer': optimizer_state,
                }, args, is_best, model_filename, scores)
        '''saving routing index'''

        model_filename = 'model_onlyrouting_{0}.pth.tar'.format(epoch)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'best_err1': best_err1,
        }, args, False, model_filename, scores)

    model_filename = 'final.pth.tar'
    saved_state = deepcopy(model.state_dict())

    optimizer_state = deepcopy(optimizer.state_dict())
    save_checkpoint({
        'epoch': args.epochs,
        'arch': args.arch,
        'state_dict': saved_state,
        'best_err1': best_err1,
        'optimizer': optimizer_state,
    }, args, 0, model_filename, scores)
    print('Best val_err1: {:.4f} at epoch {}'.format(best_err1, best_epoch))




def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Conflictloss = AverageMeter()
    avg_conflictmatrix = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()

    running_lr = None
    for i, (input, target) in enumerate(train_loader):

        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        # measure data loading time
        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, _ = model(input_var)
        if not isinstance(output, list):
            output = [output]
        losslist = []
        if not args.usingsdn:
            for j in range(len(output)):
                tmp = criterion(output[j], target_var)
                losslist.append(tmp)
        else:
            max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])  # max tau_i --- C_i values
            cur_coeffs = 0.01 + epoch * (max_coeffs / args.epochs)  # to calculate the tau at the currect epoch
            cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
            for j in range(len(output)):
                if j < (len(output) - 1):
                    tmp = float(cur_coeffs[j]) * criterion(output[j], target_var)
                else:
                    tmp = criterion(output[j], target_var)
                losslist.append(tmp)
        loss = torch.stack(losslist).sum()
        losses.update(loss.item(), input.size(0))
        # print("losslist:{0}".format(torch.stack(losslist).sum()))
        # print("loss:{0}".format(loss))
        '''Recording gradients'''
        Grad_Dictlist = []


        # measure error and record loss
        losses.update(loss.item(), input.size(0))
        # compute gradient and do SGD step


        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()


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
            losslist = []
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


def expand(w, grad):
    if grad.dim() == 2:
        w = w.view(-1, 1)
    elif grad.dim() == 4:
        w = w.view(-1, 1, 1, 1)
    elif grad.dim() == 1:
        w = w
    w_expand = w.expand_as(grad)
    return w_expand


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
            output, _ = model(input_var)
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