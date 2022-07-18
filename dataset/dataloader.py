import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import dataset.cifar_self as selfdatasets

import torch.utils.data
def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        '''modified by sy 2022.2.13'''
        train_set, validation_set = torch.utils.data.random_split(train_set, [len(train_set) - 50000, 50000],
                                                                  generator=torch.Generator().manual_seed(0))

        test_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))



    if args.data.startswith('cifar'):
        if args.use_valid:
            train_set_index = torch.randperm(len(train_set))
            # indexpath="./results/index.pth"
            if os.path.exists(os.path.join(args.save, 'index.pth')):
                print('!!!!!! Load train_set_index !!!!!!')
                train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
            else:
                print('!!!!!! Save train_set_index !!!!!!')
                torch.save(train_set_index, os.path.join(args.save, 'index.pth'))

            # if os.path.exists(indexpath):
            #     print('!!!!!! Load train_set_index !!!!!!')
            #     train_set_index = torch.load(indexpath)
            # else:
            #     print('!!!!!! Save train_set_index !!!!!!')
            #     torch.save(train_set_index, indexpath)


            if args.data.startswith('cifar'):
                num_sample_valid = 5000
            else:
                num_sample_valid = 50000
            # num_sample_valid = len(val_set)
            print("------------------------------------")
            print("split num_sample_valid: %d" % num_sample_valid)
            print("------------------------------------")

            if 'train' in args.splits:
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=args.batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[:-num_sample_valid]),
                    num_workers=args.workers, pin_memory=True)
            if 'val' in args.splits:
                val_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=args.batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[-num_sample_valid:]),
                    num_workers=args.workers, pin_memory=True)
            if 'test' in args.splits:
                test_loader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
        else:
            if 'train' in args.splits:
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            if 'val' or 'test' in args.splits:
                val_loader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
                test_loader = val_loader
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                validation_set, batch_size=args.batch_size,
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_dataloader_distributed(args):
    traindir = os.path.join(args.data_root, 'train')
    valdir = os.path.join(args.data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_set = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]))
    '''modified by sy 2022.2.13'''
    train_set,validation_set=torch.utils.data.random_split(train_set,[len(train_set)-50000,50000],generator=torch.Generator().manual_seed(0))

    test_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]))




    num_sample_valid = 50000
    # num_sample_valid = len(val_set)
    print("------------------------------------")
    print("split num_sample_valid: %d" % num_sample_valid)
    print("------------------------------------")

    sampler = torch.utils.data.distributed.DistributedSampler(train_set,num_replicas=args.numprocesspernode,rank=args.localrank)
    sampler_test = torch.utils.data.distributed.DistributedSampler(test_set)
    sampler_val = torch.utils.data.distributed.DistributedSampler(validation_set)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True,
    sampler=sampler)


    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader=torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader,sampler

def get_dataloaders_distributed_cifar(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        test_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    else:
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        test_set = datasets.CIFAR100(args.data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))

    train_set, validation_set = torch.utils.data.random_split(train_set, [len(train_set) - 5000, 5000],
                                                              generator=torch.Generator().manual_seed(0))

    sampler = torch.utils.data.distributed.DistributedSampler(train_set,num_replicas=args.numprocesspernode,rank=args.localrank)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True,
        sampler=sampler)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader,sampler


def get_dataloaders_minidatasets(args,numpersamples):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = selfdatasets.CIFAR10(args.data_root, numpersamples,train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = selfdatasets.CIFAR10(args.data_root,numpersamples, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
        num_sample_valid=int(0.1*numpersamples*10)
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = selfdatasets.CIFAR100(args.data_root,numpersamples, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = selfdatasets.CIFAR100(args.data_root, numpersamples,train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
        num_sample_valid = int(0.1 * numpersamples * 100)
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        # indexpath="./results/index.pth"
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))

        # if os.path.exists(indexpath):
        #     print('!!!!!! Load train_set_index !!!!!!')
        #     train_set_index = torch.load(indexpath)
        # else:
        #     print('!!!!!! Save train_set_index !!!!!!')
        #     torch.save(train_set_index, indexpath)


        # if args.data.startswith('cifar'):
        #     num_sample_valid = 5000
        # else:
        #     num_sample_valid = 50000
        # num_sample_valid = len(val_set)
        print("------------------------------------")
        print("split num_sample_valid: %d" % num_sample_valid)
        print("------------------------------------")

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_root",default="/home/user/users/sy/meta-fusion/")
    args=parser.parse_args()
    get_dataloader_distributed(args=args)