import numpy as np

# history_path = data_base_path + "history/hist.npy"
def load_dictionary(dictionary_path)
    return np.load(dictionary_path, allow_pickle=True).item()
