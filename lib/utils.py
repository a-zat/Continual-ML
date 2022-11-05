from tensorflow.keras.datasets import mnist

import random 
import numpy as np
import time

from lib.Kmeans_lib import k_mean_clustering
from lib.CustomLayer_lib import update_ll_OL, update_ll_CWR


'''Utility function to create train/test datasets from Mnist'''
def create_dataset(n_train, n_test):
  (data_train, label_train),(data_test, label_test) = mnist.load_data() # Load data
  
  digits_train = np.zeros((n_train,28,28))
  digits_test = np.zeros((n_test,28,28))
  label_digits_train = np.zeros(n_train)
  label_digits_test = np.zeros(n_test)

# Select random images from the dataset
  for i in range(0, n_train):
    n = random.randint(0,len(data_train)-1)
    digits_train[i,:,:] = data_train[n,:,:]
    label_digits_train[i] = label_train[n]
  for i in range(0, n_test): 
    m = random.randint(0,len(data_test)-1)
    digits_test[i,:,:] = np.copy(data_test[m,:,:])
    label_digits_test[i] = label_test[m]

  img_rows, img_cols = 28, 28
  digits_train  = digits_train.reshape(digits_train.shape[0], img_rows, img_cols, 1).astype(np.float32) / 255.0
  digits_test = digits_test.reshape(digits_test.shape[0], img_rows, img_cols, 1).astype(np.float32) / 255.0

  return digits_train, label_digits_train, digits_test, label_digits_test



'''Class to define settings used to train and test the model'''
class TrainSettings(object):
    def __init__(self):
        self.verbosity = 'SILENT'
        self.fill_cmtx = True
        self.save_output = False
        self.save_path = ''
        self.save_plots = False
        self.mode = 'UNDEFINED'
        self.datalog = [None] * 4 



'''Script to run one epoch'''
def RunOneEpoch(model, images, labels, features_saved, labels_saved):

    n_samples = images.shape[0]
    # settings.cluster_batch_size = min(settings.cluster_batch_size, len(labels))
    clust_err_array = []
    model_err_array = []

    # BATCH PROCESSING OF DATA
    n_batch = int(np.ceil(n_samples / model.clustering_batch_size))
    images_batch = np.array_split(images, n_batch)
    labels_batch = np.array_split(labels, n_batch)

    err_clu = 0 # Clustering error (entire epoch)
    err_mod = 0 # Model error (entire epoch)
    pseudolabels = []

    if model.ll_method == 'CWR':
        model_cntr = 0 
        found_digit = np.zeros(10) 
        
    for i in range(0, n_batch):
        batch_size = labels_batch[i].shape[0]# Current batch size
        print("Starting {} batch: {}/{}".format(model.settings.mode, i+1, n_batch))
        # Features extraction
        start1 = time.time()
        features_batch = model.ML_frozen.predict(images_batch[i].reshape((batch_size,28,28,1)), verbose = False)
        end1 = time.time()

        # Kmean clustering
        start2 = time.time()
        pseudolabels_batch, err_clu_batch = k_mean_clustering(features_batch, features_saved, labels_batch[i], labels_saved, model)
        end2 = time.time()
        pseudolabels.extend(pseudolabels_batch)
        err_clu += err_clu_batch
        clust_err_array.append(err_clu_batch)

        # Last Layer update
        err_mod_batch = 0
        for j in range(batch_size):
            if model.ll_method == 'OL':
                prediction = update_ll_OL(model, features_batch[j,:], pseudolabels_batch[j])
                
            if model.ll_method == 'CWR':
                
                if(model_cntr == model.update_batch_size):
                    prediction, found_digit = update_ll_CWR(model, features_batch[j,:], pseudolabels_batch[j], found_digit, True)
                    model_cntr = 0
                else:
                    prediction, found_digit = update_ll_CWR(model, features_batch[j,:], pseudolabels_batch[j], found_digit, False)
                model_cntr += 1

            if(prediction != labels_batch[i][j]):  
               err_mod_batch += 1

            # Update confusion matrix - posso creare funzione in Custom_layer
            if model.settings.fill_cmtx == True:
                for k in range(0,len(model.label)):
                    if(prediction == model.std_label[k]):
                        p = np.copy(k)
                    if(labels_batch[i][j] == model.std_label[k]):
                        t = np.copy(k)

                try:
                    t
                except NameError:
                    print("!!!VARIABLE t WAS NOT DEFINED!!!")
                    print("true label:", labels_batch[i][j])
                    print("standard labels:", model.std_label)

                else: # if variable was defined can compute..
                    model.conf_matr[t,p] += 1 
                 

        model_err_array.append(err_mod_batch)
        err_mod += err_mod_batch

        if model.settings.verbosity == 'EOBINFO':
            print("Features extraction took {:.3f} seconds and Kmean clustering took {:.3f} seconds, with {:.1%} accuracy ({} errors)".format(end1-start1, end2-start2, 1-err_clu_batch/batch_size, err_clu_batch))
            print("Batch Model errors {} ({:.1%} accuracy)".format(err_mod_batch, 1-err_mod_batch/batch_size))
    
    if model.settings.verbosity == 'EOEINFO' or model.settings.verbosity == 'EOBINFO':
        print("Total clustering error: {:.1%} ({}/{} errors, {:.1%} accuracy)".format(err_clu/n_samples, err_clu, n_samples, 1-err_clu/n_samples))
        print("Total model error: {:.1%} ({}/{} errors, {:.1%} accuracy)".format(err_mod/n_samples, err_mod, n_samples, 1-err_mod/n_samples))

    # return clust_err_array, model_err_array
    if model.settings.save_output == True:
        model.settings.datalog = list([err_clu, err_mod, clust_err_array, model_err_array])