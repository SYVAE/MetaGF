#['vgg16','resnet56','wideresnet32_4','mobilenet']
cd ../../

save_dir="/home/user/ECCV_version_MetaGF/results100/vgg16meta_raw/2022-7-8-10-29-28/"
python Test.py \
--data-root "./data/cifar/" \
--save $save_dir \
--data cifar100 \
--task cifar100 \
--gpu 0 \
--sdnarch "vgg16" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--use-valid \
--eval "anytime" \
-j 1

#save_dir="/home/user/ECCV_addedExpe/Shallow-Deep-Networks_MetaGF/results_cifar100/vgg/vgg16_sdn_cagrad/2022-5-26-9-38-13/"
#python Test.py \
#--data-root "./data/cifar/" \
#--save $save_dir \
#--data cifar100 \
#--task cifar100 \
#--gpu 0 \
#--sdnarch "vgg16" \
#--batch-size 64 \
#--epochs 100 \
#--seed 0 \
#--use-valid \
#--eval "anytime" \
#-j 1
#
#save_dir="/home/user/ECCV_addedExpe/Shallow-Deep-Networks_MetaGF/results_cifar100/vgg/vgg16_sdn_Pcgrad/2022-5-26-1-13-6/"
#python Test.py \
#--data-root "./data/cifar/" \
#--save $save_dir \
#--data cifar100 \
#--task cifar100 \
#--gpu 0 \
#--sdnarch "vgg16" \
#--batch-size 64 \
#--epochs 100 \
#--seed 0 \
#--use-valid \
#--eval "anytime" \
#-j 1
#
#save_dir="/home/user/ECCV_addedExpe/Shallow-Deep-Networks_MetaGF/results_cifar100/vgg/vgg16_sdn_metaGF/t52022-5-26-1-17-37/"
#python Test.py \
#--data-root "./data/cifar/" \
#--save $save_dir \
#--data cifar100 \
#--task cifar100 \
#--gpu 0 \
#--sdnarch "vgg16" \
#--batch-size 64 \
#--epochs 100 \
#--seed 0 \
#--use-valid \
#--eval "anytime" \
#-j 1

