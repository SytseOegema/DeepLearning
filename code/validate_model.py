from datasets import load_dataset
from transformers import DefaultDataCollator, create_optimizer, TFAutoModelForQuestionAnswering, AutoConfig
import tensorflow as tf
from helpers.preprocessing import preprocess_function
from helpers.load_data import get_squad_data, get_squad_data_small
from helpers.store_results import store_dictionary, create_specific_folder

batch_size = 16
num_epochs = 1
pre_trained_path = "../data/1647943553369700500/pretrained_model/"
# data_base_path = "../data/"
data_base_path = "/data/s3173267/BERT/"

data = get_squad_data_small(data_base_path + "squad.dat")

tokenized_data = data.map(preprocess_function, batched=True, remove_columns=data["train"].column_names)
data_collator = DefaultDataCollator(return_tensors="tf")

tf_test_set = tokenized_data["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    dummy_labels=True,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

total_train_steps = (len(tokenized_data["train"]) // batch_size) * num_epochs
print(total_train_steps)

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=total_train_steps,
)

config = AutoConfig.from_pretrained(pre_trained_path + "config.json")

model = TFAutoModelForQuestionAnswering.from_pretrained(
    pre_trained_path + "tf_model.h5",
    config=config,
)

model.compile(optimizer=optimizer)

history = model.evaluate(x=tf_test_set)

print(history)
