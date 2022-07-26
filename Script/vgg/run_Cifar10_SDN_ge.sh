cd ../../
save_dir="./results/vgg16_ge/"
seed 0
python M0_main.py \
--data-root "./data/cifar/" \
--save $save_dir \
--usingsdn 1 \
--data cifar10 \
--task cifar10 \
--gpu 2 \
--ge 1 \
--sdnarch "vgg16" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--lr 0.1 \
--use-valid \
-j 1

