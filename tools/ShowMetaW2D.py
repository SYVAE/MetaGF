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
from models.SDN_Constructing import SDN
from tools.Train_utils import *
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
    static_name="/home/sunyi/ECCV_version_MetaGF/20220709/results/resnet56meta_raw/2022-7-8-10-25-35/save_models/best_model.pth.tar"
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

    Totalsharedlist = []
    for name in model.state_dict():
        tmp_sum=0
        for i in range(0,len(ValidNodeslist)):
            if ValidNodeslist[i].__contains__(name):
                tmp_sum+=torch.sigmoid(ValidNodeslist[i][name])
                print(ValidNodeslist[i][name])

        for i in range(0,len(ValidNodeslist)):
            if ValidNodeslist[i].__contains__(name):
                ValidNodeslist[i][name]=torch.sigmoid(ValidNodeslist[i][name])/tmp_sum

    for i in range(0,len(ValidNodeslist)):
        tmplist = []
        for name in model.state_dict():
            if ValidNodeslist[i].__contains__(name):
                tmplist.append(ValidNodeslist[i][name].detach().cpu().numpy())
            else:
                tmplist.append(0)
        Totalsharedlist.append(np.stack(tmplist))

    shared_matrix=np.stack(Totalsharedlist)
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(1, 1, 1, projection='3d', facecolor='white')
    ax.set_xlabel("Exits")
    ax.set_ylabel("Shared Paramters")
    ax.set_zlabel("Fusion Weight")

    x, y = np.meshgrid(np.linspace(1, shared_matrix.shape[0], shared_matrix.shape[0]),
                       np.linspace(1, shared_matrix.shape[1], shared_matrix.shape[1]))
    X = x.ravel()
    Y = y.ravel()

    N = shared_matrix.shape[0]
    D = shared_matrix.shape[1]

    Z = shared_matrix.transpose(1, 0).reshape(-1)
    height = np.zeros_like(Z)
    width = depth = 1
    cmap_color = plt.cm.get_cmap('winter')
    level_list = np.linspace(0, 1, 65)
    color_list = cmap_color(level_list)

    # tmpZ=(N*2*((shared_matrix-np.min(shared_matrix,axis=0,keepdims=True))/(np.max(shared_matrix,axis=0,keepdims=True)-np.min(shared_matrix,axis=0,keepdims=True)))).astype(np.int64)-1
    # tmpZ=tmpZ.transpose(1, 0).reshape(-1)
    tmpZ = (64 * ((Z - np.min(Z, axis=0, keepdims=True)) / (
            np.max(Z, axis=0, keepdims=True) - np.min(Z, axis=0, keepdims=True) + 1e-5))).astype(np.int64)
    # tmpZ = tmpZ.transpose(1, 0).reshape(-1)
    c = color_list[tmpZ, 0:4]
    # im4 = ax.plot(x, y, shared_matrix.transpose(1,0), rstride=2, cstride=2, alpha=0.6, facecolor='white',
    #                       cmap="jet")
    ax.bar3d(X, Y, height, width, depth, Z, color=c, shade=False, edgecolor="black", alpha=1)
    plt.pause(0.1)
    # plt.show()
    plt.savefig("base3D.png")
    ##imshow
    plt.figure("2D")
    plt.imshow(shared_matrix)
    plt.show()
    plt.savefig("base2D.png")
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