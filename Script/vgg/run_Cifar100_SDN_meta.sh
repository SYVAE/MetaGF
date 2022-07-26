save_dir="./results100/vgg16_sdn_meta/"
#seed 0
cd ../../
# ['vgg16','resnet56','wideresnet32_4','mobilenet']
python M4main_GFvgg.py \
--data-root "./data/cifar/" \
--save $save_dir \
--data cifar100 \
--task cifar100 \
--gpu 2 \
--usingsdn 1 \
--sdnarch "vgg16" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--lr 0.1 \
--use-valid \
-j 1 \
--temperature 0.1 \
--comparedbaseline "/home/user/ECCV_version_MetaGF/Baseline_res/results_cifar100/vgg/vgg16_ge/2022-5-26-12-43-39/scores.tsv" \
--EMAoldmomentum 0.1 \
--Metalr 0.1 \
--auxiliarylr 0.1 \

