#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --job-name=python_train_BERT

module purge
module load TensorFlow/2.5.0-fosscuda-2020b

pip install -r code/requirements.txt --user

python code/pre_train_model.py
