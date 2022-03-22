from pathlib2 import Path
import numpy as np
import time

def store_dictionary(dictionary, data_base_path):
    Path(data_base_path + "dictionary").mkdir(parents=True, exist_ok=True)
    np.save(data_base_path + "dictionary/hist.npy", dictionary)
    print("data stored to: " + data_base_path + "dictionary/hist.npy")

def create_specific_folder(base_dir):
    final_base_dir = base_dir + str(time.time_ns()) + "/"
    Path(final_base_dir).mkdir(parents=True, exist_ok=True)
    return final_base_dir
