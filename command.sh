## Experiment on learning rate
# 1e-5, 5e-5, 1e-4, 5e-4
CUDA_VISIBLE_DEVICES=0 python train_sh.py \
--name covmixer_depth3/1e-5 \
--split_type _one \
--convmix_depth 3 \
--per_scene 1 \
--batch_size 1 \
--learning_rate 1e-5 \
--batch_multiplier 1000 \
--coeff_level 2 \
--cache_feature \
--model_type covmixer \
--input_type envmap

CUDA_VISIBLE_DEVICES=1 python train_sh.py \
--name covmixer_depth3/5e-5 \
--split_type _one \
--convmix_depth 3 \
--per_scene 1 \
--batch_size 1 \
--learning_rate 5e-5 \
--batch_multiplier 1000 \
--coeff_level 2 \
--cache_feature \
--model_type covmixer \
--input_type envmap

CUDA_VISIBLE_DEVICES=2 python train_sh.py \
--name covmixer_depth3/1e-4 \
--split_type _one \
--convmix_depth 3 \
--per_scene 1 \
--batch_size 1 \
--learning_rate 1e-4 \
--batch_multiplier 1000 \
--coeff_level 2 \
--cache_feature \
--model_type covmixer \
--input_type envmap


CUDA_VISIBLE_DEVICES=3 python train_sh.py \
--name covmixer_depth3/5e-4 \
--split_type _one \
--convmix_depth 3 \
--per_scene 1 \
--batch_size 1 \
--learning_rate 5e-4 \
--batch_multiplier 1000 \
--coeff_level 2 \
--cache_feature \
--model_type covmixer \
--input_type envmap

CUDA_VISIBLE_DEVICES=0 python train_sh.py \
--name covmixer_depth3/1e-3 \
--split_type _one \
--convmix_depth 3 \
--per_scene 1 \
--batch_size 1 \
--learning_rate 1e-3 \
--batch_multiplier 1000 \
--coeff_level 2 \
--cache_feature \
--model_type covmixer \
--input_type envmap

CUDA_VISIBLE_DEVICES=1 python train_sh.py \
--name covmixer_depth3/5e-3 \
--split_type _one \
--convmix_depth 3 \
--per_scene 1 \
--batch_size 1 \
--learning_rate 5e-3 \
--batch_multiplier 1000 \
--coeff_level 2 \
--cache_feature \
--model_type covmixer \
--input_type envmap