python amc_search.py --job=train --model=vgg16 --ckpt_path=path_to_checkpoint --dataset=imagenet --data_root=path_to_data_root --preserve_ratio=0.5 --lbound=0.2 --rbound=1 --reward=acc_reward --n_calibration_batches=60 --seed=2018