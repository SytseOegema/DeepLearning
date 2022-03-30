from datasets import load_metric
from transformers import TFAutoModelForQuestionAnswering, AutoConfig, DefaultDataCollator
from helpers.validation import prepare_validation_features, postprocess_qa_predictions
from helpers.load_data import get_test_squad_data
from helpers.store_results import store_data
import numpy as np
import sys

def predict(data_base_path, storage_path, pre_trained_path):
    batch_size = 16
    test_data = get_test_squad_data(data_base_path)

    original_stdout = sys.stdout

    with open(storage_path + "samples.txt", 'w') as f:
        sys.stdout = f

        for i in range(100):
            print("sample : " + str(i))
            print("ID: " + test_data[i]["id"])
            print("Title: " + test_data[i]["title"])
            print("Context: " + test_data[i]["context"])
            print("Question: " + test_data[i]["question"])
            print("Answer: " + str(test_data[i]["answers"]) + "\n")

        sys.stdout = original_stdout

    test_features = test_data.map(
        prepare_validation_features,
        batched=True,
        remove_columns=test_data.column_names,
    )

    data_collator = DefaultDataCollator(return_tensors="tf")
    test_dataset = test_features.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    config = AutoConfig.from_pretrained(pre_trained_path + "pretrained_model/config.json")

    model = TFAutoModelForQuestionAnswering.from_pretrained(
        pre_trained_path + "pretrained_model/tf_model.h5",
        config=config,
    )

    model.compile()

    raw_predictions = model.predict(x=test_dataset)

    final_predictions = postprocess_qa_predictions(
        test_data,
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
        {"id": ex["id"], "answers": ex["answers"]} for ex in test_data
    ]


    with open(storage_path + "answers.txt", 'w') as f:
        sys.stdout = f

        for i in range(100):
            print("prediction: " + str(i))
            print(formatted_predictions[i])


        sys.stdout = original_stdout

    store_data(formatted_predictions, storage_path + "formatted_predictions")

    results = metric.compute(predictions=formatted_predictions, references=references)

    store_data(formatted_predictions, storage_path + "results")

    print("\n\nResults:\n")
    print(results)
