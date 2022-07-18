save_dir="./results100/msdnet_sdn_meta/"
cd ../../
#seed 0
python M5main_GF.py \
--data-root "./data/cifar/" \
--save $save_dir \
--data cifar100 \
--gpu 0 \
--arch "msdnet" \
--batch-size 64 \
--epochs 300 \
--nBlocks 7 \
--usingsdn 0 \
--stepmode even \
--step 2 \
--base 4 \
--grFactor 1-2-4 \
--bnFactor 1-2-4 \
--growthRate 16 \
--nChannels 16 \
--seed 0 \
--use-valid \
-j 1 \
--comparedbaseline "/home/sunyi/MetaGF_V0/results/resnet56meta_raw/Metalr0.1_Auxlr0.1_EMAold0.92022-7-16-22-50-29/scores.tsv" \
--EMAoldmomentum 0.9 \
--Metalr 0.1 \
--auxiliarylr 0.1 \







