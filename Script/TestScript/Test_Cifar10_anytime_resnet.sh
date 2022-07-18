#['vgg16','resnet56','wideresnet32_4','mobilenet']
cd ../../

save_dir="/home/sunyi/MetaGF_V0/results/resnet56meta_raw/Metalr0.1_Auxlr0.1_EMAold0.92022-7-16-22-50-29/"
python Test.py \
--data-root "./data/cifar/" \
--save $save_dir \
--data cifar10 \
--task cifar10 \
--gpu 0 \
--sdnarch "resnet56" \
--batch-size 64 \
--epochs 100 \
--seed 0 \
--use-valid \
--eval "anytime" \
-j 1
