from helpers.load_results import load_dictionary, load_array
import matplotlib.pyplot as plt

pre_trained_paths = [
    "../pre_trained_models/from_scratch_5_epoch/",
    "../pre_trained_models/from_scratch_10_epoch/",
    "../pre_trained_models/pre_trained_plus_2_epoch/",
    "../pre_trained_models/pre_trained_plus_5_epoch/"
]

storage_paths = [
    "../data/from_scratch_5_epoch/",
    "../data/from_scratch_10_epoch/",
    "../data/pre_trained_plus_2_epoch/",
    "../data/pre_trained_plus_5_epoch/"
]

scratch_5_hist = load_dictionary("../pre_trained_models/from_scratch_5_epoch/results/hist.npy")
scratch_10_hist = load_dictionary("../pre_trained_models/from_scratch_10_epoch/results/hist.npy")
pre_2_hist = load_dictionary("../pre_trained_models/pre_trained_plus_2_epoch/results/hist.npy")
pre_5_hist = load_dictionary("../pre_trained_models/pre_trained_plus_5_epoch/results/hist.npy")

scratch_loss = scratch_5_hist["loss"] + scratch_10_hist["loss"]
scratch_val_loss = scratch_5_hist["val_loss"] + scratch_10_hist["val_loss"]

pre_loss = pre_2_hist["loss"] + pre_5_hist["loss"]
pre_val_loss = pre_2_hist["val_loss"] + pre_5_hist["val_loss"]

plt.figure(figsize=(12, 4))
plt.title("Training Scores")
plt.plot(range(10), scratch_loss)
plt.plot(range(10), scratch_val_loss)
plt.plot(range(4), pre_loss)
plt.plot(range(4), pre_val_loss)
plt.legend(["Scratch training loss", "Scratch vallidation loss", "Pre-Trained training loss", "Pre-Trained vallidation loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('../results/training_scores.png')

print(scratch_loss)
print(scratch_val_loss)
print(pre_loss)
print(pre_val_loss)

storage_paths = [
    "../data/from_scratch_5_epoch/",
    "../data/from_scratch_10_epoch/",
    "../data/pre_trained_plus_2_epoch/",
    "../data/pre_trained_plus_5_epoch/"
]

for idx in range(4):
    dict = load_array(storage_paths[idx] + "/results.npy")
    print(dict)
