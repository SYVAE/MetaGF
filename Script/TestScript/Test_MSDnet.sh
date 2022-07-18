save_dir="/home/sunyi/427236467710491/"
cd ../../
python Test_MSDnet.py \
--data-root "./data/cifar/" \
--save $save_dir \
--data cifar10 \
--gpu 0 \
--arch "msdnet_ge" \
--batch-size 64 \
--epochs 300 \
--nBlocks 7 \
--stepmode even \
--step 2 \
--base 4 \
--grFactor 1-2-4 \
--bnFactor 1-2-4 \
--growthRate 16 \
--nChannels 16 \
--seed 0 \
--use-valid \
--eval "anytime" \
-j 1


