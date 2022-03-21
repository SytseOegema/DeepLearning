module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
module load git/2.14.1-GCCcore-6.4.0

python3 -m venv /data/$USER/.envs/bert_env

source /data/$USER/.envs/bert_env/bin/activate

pip install --upgrade pip
pip install --upgrade wheel

pip install transformers
pip install datasets
