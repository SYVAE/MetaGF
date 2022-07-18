#['vgg16','resnet56','wideresnet32_4','mobilenet']
cd ../../
save_dir="/home/user/ECCV_version_MetaGF/results/vgg16meta_raw/2022-7-8-10-28-25/"
source activate pytorch
python Test.py \
--data-root "./data/cifar/" \
--save $save_dir \
--data cifar10 \
--task cifar10 \
--gpu 0 \
--sdnarch "vgg16" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--use-valid \
--eval "anytime" \
-j 1


