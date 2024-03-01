CUDA_VISIBLE_DEVICES=0 python train_sh.py --name missingloss_single_image --cache_feature --batch_multiplier 1000 --split "_one"


CUDA_VISIBLE_DEVICES=3 python train_sh.py --name missingloss_single_coeff4 --cache_feature --batch_multiplier 1000 --split "_one" --coeff_level 4 --per_scene 1


CUDA_VISIBLE_DEVICES=0 python train_sh.py --name missingloss_single_coeff5 --cache_feature --batch_multiplier 1000 --split "_one" --coeff_level 5 --per_scene 1


CUDA_VISIBLE_DEVICES=0 python train_sh.py --name fit_single/simple_cnn --model_type "simple_cnn" --cache_feature --batch_multiplier 1000 --split "_one" --per_scene 1 --batch_size 1



CUDA_VISIBLE_DEVICES=1 python train_sh.py --name compare_batchnorm/with --cache_feature --batch_multiplier 10 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 1
CUDA_VISIBLE_DEVICES=2 python train_sh.py --name compare_batchnorm/without --cache_feature --batch_multiplier 10 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 1 --no_batchnorm

CUDA_VISIBLE_DEVICES=3 python train_sh.py --name compare_batchnorm/with_batchsize4 --cache_feature --batch_multiplier 40 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 4


CUDA_VISIBLE_DEVICES=3 python train_sh.py --name test/architech_old --cache_feature --batch_multiplier 40 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 4
#######################################################################
CUDA_VISIBLE_DEVICES=1 python train_sh.py --name compare2_batchnorm/with --cache_feature --batch_multiplier 10 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 1
CUDA_VISIBLE_DEVICES=2 python train_sh.py --name compare2_batchnorm/without --cache_feature --batch_multiplier 10 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 1 --no_batchnorm

CUDA_VISIBLE_DEVICES=3 python train_sh.py --name compare2_batchnorm/with_b4 --cache_feature --batch_multiplier 40 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 4
CUDA_VISIBLE_DEVICES=2 python train_sh.py --name compare2_batchnorm/without_b4 --cache_feature --batch_multiplier 10 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 4 --no_batchnorm


CUDA_VISIBLE_DEVICES=3 python train_sh.py --name compare2_batchnorm/with_notrack --cache_feature --batch_multiplier 10 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 1 --no_batchnorm_track


CUDA_VISIBLE_DEVICES=0 python train_sh.py --name runs/without_batchnorm --coeff_level 5 --batch_size 1 --no_batchnorm

CUDA_VISIBLE_DEVICES=1 python train_sh.py --name test/batchnorm_evalmode --cache_feature --batch_multiplier 10 --split "_one" --coeff_level 5 --per_scene 1 --batch_size 4


