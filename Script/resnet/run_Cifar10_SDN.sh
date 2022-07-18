save_dir="./results/resnet56_sdn/"
#seed 0
cd ../../
# ['vgg16','resnet56','wideresnet32_4','mobilenet']
python M0_main.py \
--data-root "./data/cifar/" \
--save $save_dir \
--usingsdn 1 \
--data cifar10 \
--task cifar10 \
--gpu 0 \
--sdnarch "resnet56" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--lr 0.1 \
--use-valid \
--predefinedindex "/home/user/ECCV_version_MetaGF/A0index/Cifar10/resnet/index.pth" \
-j 1
