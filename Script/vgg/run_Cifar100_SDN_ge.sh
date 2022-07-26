cd ../../
save_dir="./results100/vgg16_ge/"
seed 0
python M0_main.py \
--data-root "./data/cifar/" \
--save $save_dir \
--usingsdn 1 \
--data cifar100 \
--task cifar100 \
--gpu 6 \
--ge 1 \
--sdnarch "vgg16" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--lr 0.1 \
--use-valid \
-j 1

