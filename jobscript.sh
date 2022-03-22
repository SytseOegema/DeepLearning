#!/bin/bash
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=python_train_BERT
#SBATCH --mem=2gb

module purge
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4

pip install -r code/requirements.txt

python code/main.py

deactivate
