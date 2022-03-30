# DeepLearning

Contains a Destil Bert training procedure.

## Getting Started

### Prediction
Clone the repo. Download the pre_trained models from [here](https://sytseoegema.stackstorage.com/s/zHXsV4ns4Rsmqzje) and put the tar file inside the cloned repo. The unpack the models using the command
`tar -xzvf pre_trained_models.tar.gz`

Thereafter you can execute python script `code/2_predictions.py`. This will use the pre trained models to do predictions on the SQUAD Dataset.

### Results
Similarly you can reproduce the results by downloading an archive from
[here](https://sytseoegema.stackstorage.com/s/xEiDLwFZfGSGVxJo) and unpacking it
using the command `tar -xzvf data.tar.gz`. This yields the same data as running
`2_predictions.py` with the pre_trained models.

The data is uesd in `3_results.py` to generate graphs and latex table structures.

## Code
Inside the code folder the training procedures can be found.

- `1_pre_train_model.py` : a script that can be used to pre train the distil
BERT model. It can either be used to train an existing model or to train a model
from scratch.
- `2_predictions.py` : a script that can be used to create question answering
predictions. It uses a pre_trained model.
