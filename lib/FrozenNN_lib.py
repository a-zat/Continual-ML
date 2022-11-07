import numpy as np
import tempfile
import os
import zipfile

def testing(data, label_lett, model):

    conf_matrix = np.zeros((6,6))

    prediction = model.predict(data)
    
    for i in range(0, data.shape[0]):

        # Find the max iter for both true label and prediction
        max_i_true = int(label_lett[i])

        max_i_pred = np.argmax(prediction[i,:])

        conf_matrix[max_i_true, max_i_pred] = conf_matrix[max_i_true, max_i_pred] + 1

    return conf_matrix

class save_info:
    #define init function to initialize main variables:
    def __init__(self, batch_size, epochs, metrics, optimizer,loss):

    #define main variables you want to store and use
        self.batch_size    = batch_size
        self.epochs     = epochs
        self.metrics     = metrics
        self.optimizer  = optimizer
        self.loss  = loss

def save_summary_model(model, MODEL_PATH, info, flag):
      with open(MODEL_PATH + 'model_summary.txt', "w") as new_file:
        new_file.write("PARAMETERS SAVED FROM THE TRAINING")
        match flag:
          case "original":
            new_file.write("\n\n This model has been trained for learning the first 6 digits from the MNIST dataset, this is the ORIGINAL MODEL")
          case "frozen":
            new_file.write("\n\n This model has been trained for learning the first 6 digits from the MNIST dataset, this is the FROZEN MODEL")
           # new_file.write("\n")
           # new_file.write("\n Batch size:       " + str(info.batch_size))
           # new_file.write("\n Epochs:           " + str(info.epochs))
           # new_file.write("\n Metrics:          " + str(info.metrics))
           # new_file.write("\n Optimizer:        " + info.optimizer)
            #new_file.write("\n Loss:             " + info.loss)
            #new_file.write("\n\n")
          case _:
            raise ValueError("Unknown flag")
        model.summary(print_fn=lambda x: new_file.write(x + '\n'))
      new_file.close()


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)