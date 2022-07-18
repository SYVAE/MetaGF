import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch
import math
def load_previous_accCurve(baselineFile):
    # baselineFile = '/home/sunyi/sy/A6Gradient/IMTA/tmpsave/ge/scores.tsv'
    baseline = np.loadtxt(baselineFile, delimiter='\t', skiprows=1)
    baseline_acclist = []
    baseline_trainacclist = []
    for i in range(0, baseline.shape[0]):
        baseline_acclist.append(baseline[i, 5])
        baseline_trainacclist.append(baseline[i, 4])

    fig, ax = plt.subplots()
    # ax.step(np.array(range(0, len(baseline_acclist))), np.stack(baseline_acclist), where='post', label='post')
    ax.plot(np.array(range(0, len(baseline_acclist))), np.stack(baseline_acclist), '--', color='black', alpha=0.3)

    # ax.step(np.array(range(0, len(baseline_trainacclist))), np.stack(baseline_trainacclist), where='post', label='post')
    ax.plot(np.array(range(0, len(baseline_trainacclist))), np.stack(baseline_trainacclist), '-', color='black',
            alpha=0.3)

    ax.grid(axis='x', color='0.95')
    ax.legend(title='Parameter where:')
    ax.set_frame_on(False)
    ax.set_title('Acc')
    # plt.savefig("comparison.png")
    plt.pause(0.01)

    return ax



def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'best_model.pth.tar')
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        torch.save(state, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return



def save_checkpoint_tuning(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'best_model_tuning.pth.tar')
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        torch.save(state, best_filename)

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

def accuracy(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # res.append(100.0 - correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    try:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    except:
        for param_group in optimizer._optim.param_groups:
            param_group['lr'] = lr
    return lr