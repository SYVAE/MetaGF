cd ../../
save_dir="./results100/resnet56_ge/"
seed 0
python M0_main.py \
--data-root "./data/cifar/" \
--save $save_dir \
--usingsdn 1 \
--data cifar100 \
--task cifar100 \
--gpu 7 \
--ge 1 \
--sdnarch "resnet56" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--lr 0.1 \
--use-valid \
--predefinedindex "/home/user/ECCV_version_MetaGF/A0index/Cifar100/resnet/index.pth" \
-j 1

