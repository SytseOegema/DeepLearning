#!/bin/bash
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=python_train_BERT
#SBATCH --mem=2gb

module purge
module load TensorFlow/2.5.0-fosscuda-2020b

pip install -r code/requirements.txt --user

python code/main.py

deactivate
