save_dir="./results/resnet56meta_raw/"
#seed 0
cd ../../
# ['vgg16','resnet56','wideresnet32_4','mobilenet']
python M5main_GF.py \
--data-root "./data/cifar/" \
--save $save_dir \
--data cifar10 \
--task cifar10 \
--gpu 0 \
--ge 0 \
--sdnarch "resnet56" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--lr 0.1 \
--use-valid \
--comparedbaseline "./results/resnet56meta_raw/2022-7-12-13-49-57_weightedaux/scores.tsv" \
--EMAoldmomentum 0.9 \
--Metalr 0.1 \
--auxiliarylr 0.1 \
-j 1
