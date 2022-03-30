from datasets import load_dataset, DatasetDict
from os.path import exists
import numpy as np

# get_squad_data creates a data set for training, validation and testing.
#
# The data is stored on disk the first time to ensure the data set is the same
# over many runs.
def get_squad_data(path_to_file):
    if(exists(path_to_file)):
        return DatasetDict.load_from_disk(path_to_file)


    squad = load_dataset("squad")
    val_len = squad['validation'].shape[0]

    squad["test"] = squad["validation"].select([x for x in range(int(val_len / 2) + 1, val_len)])
    squad["validation"] = squad["validation"].select([x for x in range(int(val_len / 2))])

    squad.save_to_disk(path_to_file)

    return squad

def get_squad_data_small(path_to_file):
    squad = get_squad_data(path_to_file)

    squad["train"] = squad["train"].select([x for x in range(20)])
    squad["test"] = squad["test"].select([x for x in range(20)])
    squad["validation"] = squad["validation"].select([x for x in range(20)])

    return squad


def get_test_squad_data(path_to_file):
    squad = get_squad_data(path_to_file)
    test_data = squad["test"]
    indices = [*range(1000)]
    return test_data.select(indices)
