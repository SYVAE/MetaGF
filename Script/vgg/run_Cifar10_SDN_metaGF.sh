save_dir="./results/vgg16meta_raw/"
#seed 0
cd ../../
# ['vgg16','resnet56','wideresnet32_4','mobilenet']
python M4main_GFvgg.py \
--data-root "./data/cifar/" \
--save $save_dir \
--data cifar10 \
--task cifar10 \
--gpu 1 \
--usingsdn 1 \
--sdnarch "vgg16" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--lr 0.1 \
--use-valid \
-j 1 \
--temperature 1 \
--comparedbaseline "/home/user/ECCV_version_MetaGF/Baseline_res/results/vgg/vgg16_ge/2022-5-26-10-55-58/scores.tsv" \
--EMAoldmomentum 0.1 \
--Metalr 0.001 \
--auxiliarylr 0.1 \
