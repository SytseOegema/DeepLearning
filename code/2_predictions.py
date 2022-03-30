from helpers.store_results import create_specific_folder
from helpers.predict import predict

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

data_base_path = "../data/squad.dat"

for idx in range(4):
    create_specific_folder(storage_paths[idx])
    print("pre_trained_path: " + pre_trained_paths[idx])
    print("storage_path: " + storage_paths[idx])
    print("data_base_path: " + data_base_path)

    predict(data_base_path, storage_paths[idx], pre_trained_paths[idx])
