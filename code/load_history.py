import numpy as np

# history_path = data_base_path + "history/hist.npy"
def load_history(history_path)
    return np.load(history_path, allow_pickle=True).item()
