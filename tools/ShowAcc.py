import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def Show():
    baselineFile ='/home/sunyi/ECCV_version_MetaGF/results/resnet56meta_raw/2022-7-11-9-47-47_aux0.01/scores.tsv'
    baseline = np.loadtxt(baselineFile, delimiter='\t', skiprows=1)
    acclist=[]
    trainacclist=[]
    for i in range(0,baseline.shape[0]):
        acclist.append(baseline[i,5])
        trainacclist.append(baseline[i,4])

    fig, ax = plt.subplots()
    # ax.step(np.array(range(0,len(acclist))), np.stack(acclist), where='post', label='post')
    ax.plot(np.array(range(0,len(acclist))), np.stack(acclist), '--', color='black', alpha=0.3,label='nopcgrad')

    # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
    ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), '-', color='black', alpha=0.3,label='nopcgrad')

    baselineFile = '/home/sunyi/ECCV_version_MetaGF/results/resnet56meta_raw/2022-7-11-11-2-29/scores.tsv'
    baseline = np.loadtxt(baselineFile, delimiter='\t', skiprows=1)
    acclist = []
    trainacclist = []
    for i in range(0, baseline.shape[0]):
        acclist.append(baseline[i, 5])
        trainacclist.append(baseline[i, 4])

    ax.plot(np.array(range(0, len(acclist))), np.stack(acclist), '--', color='red', alpha=0.3)

    # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
    ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), '-', color='red', alpha=0.3)


    ax.grid(axis='x', color='0.95')
    ax.legend(title='Parameter where:')
    ax.set_frame_on(False)
    ax.set_title('matplotlib.axes.Axes.set_frame_on() Example')
    plt.savefig("comparison.png")
    plt.show()

def Show_List():
    namelist=['ge','cagrad','pcgrad','meta','sdn']
    color = ['cyan', 'green', 'blue', 'red', 'black', 'yellow']
    root='/home/sunyi/ECCV_version_MetaGF/Baseline_res/results100/vgg/'
    fig, ax = plt.subplots()

    count=0
    for name in namelist:
        baselineFile =root+'/'+name+'/1/scores.tsv'
        if not os.path.exists(baselineFile):
            assert("no such file:{0}".format(baselineFile))
        baseline = np.loadtxt(baselineFile, delimiter='\t', skiprows=1)
        print(baselineFile)
        print(len(baseline))
        acclist=[]
        trainacclist=[]
        for i in range(0,100):
            acclist.append(baseline[i,5])
            trainacclist.append(baseline[i,4])


        # ax.step(np.array(range(0,len(acclist))), np.stack(acclist), where='post', label='post')
        # ax.plot(np.array(range(0,len(acclist))), np.stack(acclist), '--', color=color[count], alpha=0.3,label=name)

        # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
        ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), '-', color=color[count], alpha=0.3,label=name)
        count+=1

    ax.grid(axis='x', color='0.95')
    ax.legend(title='Method:')
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel('Classification Accuracy(%)', fontsize=15)
    ax.set_axis_on()
    ax.set_frame_on(False)
    plt.savefig(root+"/comparison_convergence.png")
    plt.show()


if __name__ == '__main__':
    # Show()
    Show_List()