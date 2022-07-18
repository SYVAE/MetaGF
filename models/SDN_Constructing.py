import copy
import torch
import time
import os
import random
import numpy as np
import torch.nn as nn
import aux_funcs  as af
import network_architectures as arcs

from architectures.CNNs.VGG import VGG

def SDN(args):

    if args.sdnarch=='vgg16':

        model=arcs.create_vgg16bn(args.task,args.ge)
        args.nBlocks=7
        args.weight_decay= 0.0005
    elif args.sdnarch=='resnet56':
        model=arcs.create_resnet56( args.task, args.ge)
        args.nBlocks = 7
        args.weight_decay = 0.0001
    elif args.sdnarch == 'wideresnet32_4':
        model=arcs.create_wideresnet32_4( args.task,args.ge)
        args.nBlocks = 7
        args.weight_decay = 0.0005
    elif args.sdnarch == 'mobilenet':
        model=arcs.create_mobilenet( args.task, args.ge)
        args.weight_decay = 0.0001
        args.nBlocks = 7
    else:
        raise("no such kind of model:{0}".format(args.sdnarch))

    return model
