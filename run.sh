#!/bin/bash

#SBATCH --job-name=pretain
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=5-00:00:00
#SBATCH --nodelist=xxx
#SBATCH -p a6000
#SBATCH -N 1
#SBATCH --output=nki_keyword_seg_%A.out
#SBATCH --error=nki_keyword_seg_%A.err


nvidia-smi

echo "starting training"
#pretrain
##nohup python r'/run.py -c r'/pretrain_config.yaml --train
##nohup python r'/run.py -c r'/pretrain_config_test.yaml --test

#cls
#python r'/run.py  -c r'/classification_config_1.yaml --train --test --train_pct 1 &
#python r'/run.py  -c r'/classification_config_0.1.yaml --train --test --train_pct 0.1 &
#python r'/run.py  -c r'/classification_config_0.01.yaml --train --test --train_pct 0.01

#seg
#python r'/run.py  -c r'/segmentation_config_1.yaml --train --test --train_pct 1 &
#python r'/run.py  -c r'/segmentation_config_0.1.yaml --train --test --train_pct 0.1 &
#python r'/run.py  -c r'/segmentation_config_0.01.yaml --train --test --train_pct 0.01

