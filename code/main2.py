import numpy as np
data_base_path = "./data/"

d2 = np.load(data_base_path + "history/hist.npy", allow_pickle=True).item()
print(d2)
print(d2.keys())
