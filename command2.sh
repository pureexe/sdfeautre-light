## Experiment on learning rate
# 1e-5, 5e-5, 1e-4, 5e-4
CUDA_VISIBLE_DEVICES=0 python train_sh.py \
--name covmixer_full_depth3_order2/1e-5 \
--dataset /data2/pakkapon/relight/sdfeautre-light/data/polyhaven \
--convmix_depth 3 \
--learning_rate 1e-5 \
--coeff_level 2 \
--model_type covmixer \
--input_type envmap

CUDA_VISIBLE_DEVICES=1 python train_sh.py \
--name covmixer_full_depth3_order2/5e-5 \
--dataset /data2/pakkapon/relight/sdfeautre-light/data/polyhaven \
--convmix_depth 3 \
--learning_rate 5e-5 \
--coeff_level 2 \
--model_type covmixer \
--input_type envmap

CUDA_VISIBLE_DEVICES=2 python train_sh.py \
--name covmixer_full_depth3_order2/1e-4 \
--dataset /data2/pakkapon/relight/sdfeautre-light/data/polyhaven \
--convmix_depth 3 \
--learning_rate 1e-4 \
--coeff_level 2 \
--model_type covmixer \
--input_type envmap


CUDA_VISIBLE_DEVICES=3 python train_sh.py \
--name covmixer_full_depth3_order2/5e-4 \
--dataset /data2/pakkapon/relight/sdfeautre-light/data/polyhaven \
--convmix_depth 3 \
--learning_rate 5e-4 \
--coeff_level 2 \
--model_type covmixer \
--input_type envmap

CUDA_VISIBLE_DEVICES=0 python train_sh.py \
--name covmixer_full_depth3_order2/1e-3 \
--convmix_depth 3 \
--learning_rate 1e-3 \
--coeff_level 2 \
--model_type covmixer \
--input_type envmap

CUDA_VISIBLE_DEVICES=1 python train_sh.py \
--name covmixer_full_depth3_order2/5e-3 \
--convmix_depth 3 \
--learning_rate 5e-3 \
--coeff_level 2 \
--model_type covmixer \
--input_type envmap