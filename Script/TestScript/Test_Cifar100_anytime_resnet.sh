#['vgg16','resnet56','wideresnet32_4','mobilenet']
cd ../../


save_dir="/home/user/ECCV_version_MetaGF/results100/resnet56meta_raw/2022-7-8-10-15-52/"
python Test.py \
--data-root "./data/cifar/" \
--save $save_dir \
--data cifar100 \
--task cifar100 \
--gpu 0 \
--sdnarch "resnet56" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--use-valid \
--eval "anytime" \
-j 1



#save_dir="/home/user/ECCV_addedExpe/Shallow-Deep-Networks_MetaGF/results_cifar100/resnet/resnet56_sdn_cagrad/2022-5-26-12-56-35/"
#python Test.py \
#--data-root "./data/cifar/" \
#--save $save_dir \
#--data cifar100 \
#--task cifar100 \
#--gpu 0 \
#--sdnarch "resnet56" \
#--batch-size 64 \
#--epochs 100 \
#--seed 0 \
#--use-valid \
#--eval "anytime" \
#-j 1
#
#
#
#save_dir="/home/user/ECCV_addedExpe/Shallow-Deep-Networks_MetaGF/results_cifar100/resnet/resnet56_sdn_Pcgrad/2022-5-26-5-35-42/"
#python Test.py \
#--data-root "./data/cifar/" \
#--save $save_dir \
#--data cifar100 \
#--task cifar100 \
#--gpu 0 \
#--sdnarch "resnet56" \
#--batch-size 64 \
#--epochs 100 \
#--seed 0 \
#--use-valid \
#--eval "anytime" \
#-j 1


