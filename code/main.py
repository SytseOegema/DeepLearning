from datasets import load_dataset
from transformers import DefaultDataCollator, create_optimizer, TFAutoModelForQuestionAnswering, DistilBertConfig
import tensorflow as tf
from preprocessing import preprocess_function
from load_data import get_squad_data, get_squad_data_small
import numpy as np

# data_base_path = "../data/"
data_base_path = "/data/s3173267/BERT/"

data = get_squad_data_small(data_base_path + "squad.dat")

tokenized_data = data.map(preprocess_function, batched=True, remove_columns=data["train"].column_names)
data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_set = tokenized_data["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    dummy_labels=True,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = tokenized_data["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    dummy_labels=True,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

batch_size = 16
num_epochs = 1
total_train_steps = (len(tokenized_data["train"]) // batch_size) * num_epochs
print(total_train_steps)

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=total_train_steps,
)

config = DistilBertConfig()

model = TFAutoModelForQuestionAnswering.from_config(config)

model.compile(optimizer=optimizer)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=data_base_path + "checkpoints/cp-{epoch:04d}.ckpt",
    verbose=1,
    save_weights_only=True,
    save_freq=1*batch_size)

history = model.fit(
    x=tf_train_set,
    validation_data=tf_validation_set,
    callbacks=[cp_callback],
    validation_freq=1,
    epochs=num_epochs)

print(history.history.keys())

np.save(data_base_path + "history/hist.npy", history.history)
