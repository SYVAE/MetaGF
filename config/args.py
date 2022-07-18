import os
import glob
import time
import argparse

# model_names = list(map(lambda n: os.path.basename(n)[:-3],
#                        glob.glob('models/[A-Za-z]*.py')))
model_names = ['msdnet', 'msdnet_ge', 'IMTA_MSDNet', 'mobilenet_imagenet','msdnet_ge_gradient']

arg_parser = argparse.ArgumentParser(
                description='Image classification PK main script')


sdn_exp=arg_parser.add_argument_group('SDN','ConstructingSDN')
sdn_exp.add_argument('--sdnarch',default='resnet56',type=str,choices=['vgg16','resnet56','wideresnet32_4','mobilenet'])
sdn_exp.add_argument('--task',default='cifar10',type=str,choices=['cifar10','cifar100','tinyimagenet'])
sdn_exp.add_argument('--ge',default=0,type=int)
sdn_exp.add_argument('--usingsdn',default=1,type=int)
sdn_exp.add_argument('--predefinedindex',default='index.pth',type=str)

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--model_filename',default="tmp.pth",type=str)
exp_group.add_argument('--save', default='tmpsave/',
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')

###----parameter
exp_group.add_argument('--use_metaGF',default=1,type=int)
exp_group.add_argument('--temperature',default=1,type=float)
exp_group.add_argument('--comparedbaseline',default="score.tsv",type=str)
exp_group.add_argument("--EMAoldmomentum",default=0.9,type=float)
exp_group.add_argument("--Metalr",default=0.1,type=float)
exp_group.add_argument("--auxiliarylr",default=0.1,type=float)# the adjust scale of the learning rate in the auxiliary training







exp_group.add_argument('--numprocesspernode',default=1,type=int,help="num of gpu")
exp_group.add_argument('--localrank',default=0,type=int,help="rank of process in the node")
exp_group.add_argument('--masteraddress',default='127.0.0.1',type=str)
exp_group.add_argument('--resume', action='store_true',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--eval', '--evaluate', dest='evalmode', default=None,
                       choices=['anytime', 'dynamic'],
                       help='way to evaluate')
exp_group.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu',default="0",
                    help='GPU available.')
exp_group.add_argument('--KD',default=0,type=int,required=False)

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet'],
                        help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='./data/cifar/',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true',
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')


# model arch related
arch_group = arg_parser.add_argument_group('arch',
                                           'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='msdnet_ge',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
arch_group.add_argument('-d', '--depth', default=56, type=int, metavar='D',
                        help='depth (default=56)')
arch_group.add_argument('--drop-rate', default=0.0, type=float,
                        metavar='DROPRATE', help='dropout rate (default: 0.2)')
arch_group.add_argument('--death-mode', default='none',
                        choices=['none', 'linear', 'uniform'],
                        help='death mode (default: none)')
arch_group.add_argument('--death-rate', default=0.5, type=float,
                        help='death rate rate (default: 0.5)')
# arch_group.add_argument('--growth-rate', default=12, type=int,
#                         metavar='GR', help='Growth rate of DenseNet'
#                         ' (1 means dot\'t use compression) (default: 0.5)')
arch_group.add_argument('--bn-size', default=4, type=int,
                        metavar='B', help='bottle neck ratio of DenseNet'
                        ' (0 means dot\'t use bottle necks) (default: 4)')
arch_group.add_argument('--reduction', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')
# used to set the argument when to resume automatically
arch_resume_names = ['arch', 'depth', 'death_mode', 'death_rate', 'death_rate',
                     'growth_rate', 'bn_size', 'compression']

# msdnet config
arch_group.add_argument('--nBlocks', type=int, default=5)
arch_group.add_argument('--nChannels', type=int, default=32)
arch_group.add_argument('--base', type=int,default=4)
arch_group.add_argument('--stepmode', type=str, default='even',choices=['even', 'lin_grow'])
arch_group.add_argument('--step', type=int, default=1)
arch_group.add_argument('--growthRate', type=int, default=6)
arch_group.add_argument('--grFactor', default='1-2-4', type=str)
arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
arch_group.add_argument('--bnFactor', default='1-2-4')
arch_group.add_argument('--bottleneck', default=True, type=bool)
arch_group.add_argument('--pretrained', default=None, type=str, metavar='PATH',
                       help='path to load pretrained msdnet (default: none)')
arch_group.add_argument('--priornet', default=None, type=str, metavar='PATH',
                       help='path to load pretrained priornet (default: none)')


# training related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')
optim_group.add_argument('--trainer', default='train', type=str,
                         help='trainer file name without ".py"'
                         ' (default: train)')
optim_group.add_argument('--epochs', default=100, type=int, metavar='N',
                         help='number of total epochs to run (default: 164)')
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('--switch-mode', default=300, type=int, metavar='N',
                         help='number of epochs to switch mode (default: 300)')
optim_group.add_argument('--patience', default=0, type=int, metavar='N',
                         help='patience for early stopping'
                         '(0 means no early stopping)')
optim_group.add_argument('-b', '--batch-size', default=64, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr-type', default='multistep', type=str, metavar='T',
                        help='learning rate strategy (default: multistep)',
                        choices=['cosine', 'multistep'])
optim_group.add_argument('--decay-rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
optim_group.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
optim_group.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')

### add kd hyperparameters

optim_group.add_argument('--gamma', default=0.9, type=float, metavar='M',
                         help='gamma for kld loss')

optim_group.add_argument('-T', default=3.0, type=float, metavar='M',
                         help='Temperature for KD')

