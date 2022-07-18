save_dir="./results/msdnet_sdn/"
#seed 0
cd ../../
# ['vgg16','resnet56','wideresnet32_4','mobilenet']
python M0_main.py \
--data-root "./data/cifar/" \
--save $save_dir \
--usingsdn 0 \
--data cifar10 \
--task cifar10 \
--gpu 0 \
--batch-size 64 \
--seed 0 \
--lr 0.1 \
--use-valid \
-j 1 \
--arch "msdnet" \
--epochs 300 \
--nBlocks 7 \
--stepmode even \
--step 2 \
--base 4 \
--grFactor 1-2-4 \
--bnFactor 1-2-4 \
--growthRate 16 \
--nChannels 16 \
