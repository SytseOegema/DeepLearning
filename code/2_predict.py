from datasets import load_metric
from transformers import TFAutoModelForQuestionAnswering, AutoConfig, DefaultDataCollator
from helpers.validation import prepare_validation_features, postprocess_qa_predictions
from helpers.load_data import get_squad_data, get_squad_data_small
from helpers.store_results import store_data
import numpy as np

pre_trained_path = "../data/1648121878523403500/pretrained_model/"
storage_path = "../data/predictions/"
data_base_path = "../data"
batch_size = 16

datasets = get_squad_data_small(data_base_path + "squad.dat")

print("\n\nSamples:")
for i in range(10):
    print("sample : " + str(i))
    print("ID: " + datasets["test"][i]["id"])
    print("Title: " + datasets["test"][i]["title"])
    print("Context: " + datasets["test"][i]["context"])
    print("Question: " + datasets["test"][i]["question"])
    print("Answer: " + str(datasets["test"][i]["answers"]))

print("\n\n")

test_features = datasets["test"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=datasets["test"].column_names,
)

data_collator = DefaultDataCollator(return_tensors="tf")
test_dataset = test_features.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)

config = AutoConfig.from_pretrained(pre_trained_path + "config.json")

model = TFAutoModelForQuestionAnswering.from_pretrained(
    pre_trained_path + "tf_model.h5",
    config=config,
)

model.compile()

raw_predictions = model.predict(x=test_dataset)

final_predictions = postprocess_qa_predictions(
    datasets["test"],
    test_features,
    raw_predictions["start_logits"],
    raw_predictions["end_logits"]
)

metric = load_metric("squad_v2")

formatted_predictions = [
    {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
    for k, v in final_predictions.items()
]

references = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in datasets["test"]
]

print("\n\nFormatted Predictions:")
for i in range(10):
    print("prediction: " + str(i))
    print(formatted_predictions[i])

store_data(formatted_predictions, storage_path + "formatted_predictions")

results = metric.compute(predictions=formatted_predictions, references=references)

store_data(formatted_predictions, storage_path + "results")

print("\n\nResults:\n")
print(results)
