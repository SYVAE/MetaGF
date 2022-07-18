import torch
import numpy as np
import  torch.nn as nn
import matplotlib.pyplot as plt
import models
from config.args import arg_parser, arch_resume_names
from tools.opcounter import measure_model
from models.adaptive_inference import dynamic_evaluate
import models
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from tools.Train_utils import *
from models.SDN_Constructing import SDN
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

def sy_load_static():
    static_name = "/home/sunyi/ECCV_version_MetaGF/results/resnet56meta_raw/2022-7-11-10-37-54/save_models/best_model.pth.tar"
    # ['vgg16','resnet56','wideresnet32_4','mobilenet']
    args.sdnarch = 'resnet56'
    args.data = 'cifar10'
    args.task = args.data

    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000

    dict_model = torch.load(static_name, map_location="cuda:0")
    model = SDN(args)

    ValidNodeslist = dict_model['routing_index']
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    plt.figure("weight distribution")
    plt.clf()

    tmplist = []
    # for name in model.state_dict():
    #     tmp_sum=0
    #     for i in range(0,len(ValidNodeslist)):
    #         if ValidNodeslist[i].__contains__(name):
    #             tmp_sum+=torch.sigmoid(ValidNodeslist[i][name])
    #             print(ValidNodeslist[i][name])
    #
    #     for i in range(0,len(ValidNodeslist)):
    #         if ValidNodeslist[i].__contains__(name):
    #             ValidNodeslist[i][name]=torch.sigmoid(ValidNodeslist[i][name])/tmp_sum
    # for i in range(0,len(ValidNodeslist)):
    #     tmplist = []
    #     for name in model.state_dict():
    #         if ValidNodeslist[i].__contains__(name):
    #             tmplist.append(ValidNodeslist[i][name].detach().cpu().numpy())
    #         else:
    #             tmplist.append(0)
    #     plt.subplot(3, len(ValidNodeslist) // 3 + 1, i + 1)
    #     # plt.plot(np.array(range(0, len(tmplist))), tmplist)
    #     plt.bar(np.array(range(0, len(tmplist))),height=tmplist)
    #     print(np.mean(np.stack(tmplist)))

    for name ,p in model.named_parameters():
        tmp_sum = 0
        print(p.dim())
        if p.dim()==1:
            continue
        for i in range(0, len(ValidNodeslist)):
            if ValidNodeslist[i].__contains__(name):
                tmp_sum += torch.sigmoid(ValidNodeslist[i][name])
                print(ValidNodeslist[i][name])

        for i in range(0, len(ValidNodeslist)):
            if ValidNodeslist[i].__contains__(name):
                ValidNodeslist[i][name] = torch.sigmoid(ValidNodeslist[i][name]) / tmp_sum


    for i in range(0,len(ValidNodeslist)):
        tmplist = []
        for name ,p in model.named_parameters():
            if p.dim() == 1:
                continue
            if ValidNodeslist[i].__contains__(name):
                tmplist.append(ValidNodeslist[i][name].detach().cpu().numpy())
            else:
                tmplist.append(0)
        plt.subplot(3, len(ValidNodeslist) // 3 + 1, i + 1)
        # plt.plot(np.array(range(0, len(tmplist))), tmplist)
        plt.bar(np.array(range(0, len(tmplist))),height=tmplist)
        print(np.mean(np.stack(tmplist)))
    plt.show()


def checking_sampleindex():
    index=torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/setting1/GE100/1/index.pth'))
    print(index)
    index = torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/reptile100_meta/1678536634327/index.pth'))
    print(index)
    index = torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/setting1/GE100_pcgrad/1/index.pth'))
    print(index)
    index = torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/setting1/MSDnet/1/index.pth'))
    print(index)
    index = torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/setting1/reptile100_meta_fineclassifier/1/index.pth'))
    print(index)


if __name__ == '__main__':
    sy_load_static()
    # checking_sampleindex()