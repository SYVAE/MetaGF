import numpy as np
import matplotlib.pyplot as plt
import torch
color=['cyan','green','blue','red','black','yellow']
def comparison():

    baselinepath="/home/sunyi/Meta_Fusion_remote/A0_CifarTrainCode/results/Cifar100_meta_routing/2022-5-16-6-50-5/"
    reptilepath="/home/sunyi/Meta_Fusion_remote/A0_CifarTrainCode/results/Cifar100_meta_routing_t/t10.02022-5-18-0-47-57/"

    file = baselinepath + "/dynamic.txt"
    MSDNet_dynamic_performance = np.loadtxt(file)
    MSDNet_dynamic_performance[:, 1] /= 1e6

    file = reptilepath + "/dynamic.txt"
    Meta_dynamic_performance = np.loadtxt(file)
    Meta_dynamic_performance[:, 1] /= 1e6

    VIB_performance =Meta_dynamic_performance
    fig, ax = plt.subplots()
    # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
    ax.plot(MSDNet_dynamic_performance[:, 1], MSDNet_dynamic_performance[:, 0], '--', color='black', alpha=0.3,label="ge")

    # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
    ax.plot(VIB_performance[:, 1], VIB_performance[:, 0], '-', color='r', alpha=0.3,label="meta")
    ax.grid(axis='x', color='0.95')
    ax.legend(title='Parameter where:')
    ax.set_frame_on(False)
    ax.set_title('GEvsMEta')
    plt.savefig("comparison.png")
    plt.show()



def comparison_list():
    fig, ax = plt.subplots()
    folder="/home/sunyi/ECCV_version_MetaGF/Baseline_res/results100/resnet"
    for i in range(1):
        # i=1
        baselinepath =folder+ "/meta/" + str(i + 1)

        file = baselinepath + "/dynamic.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        MSDNet_dynamic_performance[:, 1] /= 1e6

        # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
        ax.plot(MSDNet_dynamic_performance[:, 1], MSDNet_dynamic_performance[:, 0], '-', color=color[3], marker=marker[3],label="meta-Fusion",lw=2)


    for i in range(1):
        # i=1
        baselinepath=folder+ "/ge/"+str(i+1)

        file = baselinepath + "/dynamic.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        MSDNet_dynamic_performance[:, 1] /= 1e6

        # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
        ax.plot(MSDNet_dynamic_performance[:, 1], MSDNet_dynamic_performance[:, 0], '-', color=color[0],marker=marker[0],label="ge",lw=2)

    for i in range(1):
        # i=1
        baselinepath = folder+ "/pcgrad/" + str(i + 1)

        file = baselinepath + "/dynamic.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        MSDNet_dynamic_performance[:, 1] /= 1e6

        # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
        ax.plot(MSDNet_dynamic_performance[:, 1], MSDNet_dynamic_performance[:, 0], '-', color=color[1],marker=marker[1], label="pcgrad",lw=2)

    for i in range(1):
        # i=1
        baselinepath = folder+ "/cagrad/" + str(i + 1)

        file = baselinepath + "/dynamic.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        MSDNet_dynamic_performance[:, 1] /= 1e6

        # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
        ax.plot(MSDNet_dynamic_performance[:, 1], MSDNet_dynamic_performance[:, 0], '-', color=color[2],marker=marker[2], label="cagrad",lw=2 )

    for i in range(1):
        # i=1
        baselinepath =folder+  "/sdn/" + str(i + 1)

        file = baselinepath + "/dynamic.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        MSDNet_dynamic_performance[:, 1] /= 1e6

        # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
        ax.plot(MSDNet_dynamic_performance[:, 1], MSDNet_dynamic_performance[:, 0], '-', color=color[4],marker=marker[4], label="baseline",lw=2)
    ax.grid(True, linestyle='-.')
    # ax.grid(axis='x', color='1')
    # ax.grid(axis='y', color='1')
    ax.legend(title='Method:')
    ax.set_title('Budgeted batch classification on CIFAR-100')
    ax.set_xlabel('Flops(M)',fontsize=20)
    ax.set_ylabel('Classification Accuracy(%)',fontsize=20)
    ax.set_axis_on()
    ax.set_frame_on(False)
    plt.savefig(folder+"/comparison.png")
    plt.show()


# marker= ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd','.', ',']
marker= ['.', '.', '.', '.', '.', '<', '>', 's', 'd','.', ',']
def comparison_list_anytime():
    '''Why top-1 accuracy '''

    fig, ax = plt.subplots()
    compareid=0

    folder="/home/sunyi/sy/meta_fusion/Meta_Fusion/results/setting1"
    for i in range(1):
        # i=1
        baselinepath = folder+"/reptile100_meta_fineclassifier/" + str(i + 1)

        file = baselinepath + "/anytime.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        # MSDNet_dynamic_performance[:, 1] /= 1e6

        ax.step(range(0,MSDNet_dynamic_performance.shape[0]),MSDNet_dynamic_performance[:,compareid],  color=color[3], marker=marker[3],label="meta-Fusion",lw=2)
        # ax.plot(np.array(range(0,MSDNet_dynamic_performance.shape[0])), MSDNet_dynamic_performance[:, compareid], '-', color=color[3], alpha=0.3,
        #         label="meta-Fusion" + str(i))
    for i in range(1):
        # i=1
        baselinepath=folder+"/GE100/"+str(i+1)

        file = baselinepath + "/anytime.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        # MSDNet_dynamic_performance[:, 1] /= 1e6

        # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
        ax.step(np.array(range(0,MSDNet_dynamic_performance.shape[0])), MSDNet_dynamic_performance[:, compareid], '-', color=color[0], marker=marker[0],label="ge",lw=2)

    for i in range(1):
        # i=1
        baselinepath = folder+"/GE100_pcgrad/" + str(i + 1)

        file = baselinepath + "/anytime.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        # MSDNet_dynamic_performance[:, 1] /= 1e6

        # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
        ax.step(np.array(range(0,MSDNet_dynamic_performance.shape[0])), MSDNet_dynamic_performance[:,compareid], '-', color=color[1],marker=marker[1],
                label="pcgrad",lw=2)

    for i in range(1):
        # i=1
        baselinepath = folder+"/GE100cacgrad/" + str(i + 1)

        file = baselinepath + "/anytime.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        # MSDNet_dynamic_performance[:, 1] /= 1e6

        # ax.step(MSDNet_dynamic_performance[:,1],MSDNet_dynamic_performance[:,0], where='post', label='post')
        ax.step(np.array(range(0,MSDNet_dynamic_performance.shape[0])), MSDNet_dynamic_performance[:, compareid], '-', color=color[2],marker=marker[2],
                label="cagrad" ,lw=2)

    for i in range(1):
        # i=1
        baselinepath = folder+"/MSDnet/" + str(i + 1)

        file = baselinepath + "/anytime.txt"
        MSDNet_dynamic_performance = np.loadtxt(file)
        # MSDNet_dynamic_performance[:, 1] /= 1e6

        ax.step(np.array(range(0,MSDNet_dynamic_performance.shape[0])), MSDNet_dynamic_performance[:, compareid], '-', color=color[4], marker=marker[4],
                label="baseline" ,lw=2)

    ax.grid(True, linestyle='-.')
    # ax.grid(axis='x', color='0.95')

    ax.legend(title='Method:')
    ax.set_frame_on(False)
    ax.set_title('Anytime prediction on CIFAR-100')
    ax.set_xlabel('Exit')
    ax.set_ylabel('Acc')
    ax.set_axis_on()
    plt.savefig("comparison.png")
    plt.show()


if __name__ == '__main__':
    # comparison()
    comparison_list()
    # comparison_list_anytime()