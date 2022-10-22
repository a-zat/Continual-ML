from sklearn.cluster import KMeans
from tensorflow import keras

#import random 
#from random import seed

import time
import numpy as np
from lib.EvalMetrics import *

#def cluster_label_count(clusters, labels):
 #   count = {}

    # Get unique clusters and labels
#    unique_clusters = list(set(clusters))
#    unique_labels = list(set(labels))

    # Create counter for each cluster/label combination and set it to 0
 #   for cluster in unique_clusters:
 #       count[cluster] = {}

 #       for label in unique_labels:
 #           count[cluster][label] = 0
#
    # Let's count
#    for i in range(len(clusters)):
 #       count[clusters[i]][labels[i]] += 1

 #   cluster_df = pd.DataFrame(count)

 #   return cluster_df, count

'''Function to compute softmax of the custom layer'''
def softmax(array):
    
    if(len(array.shape)==2):
        array = array[0]
        
    size    = len(array)
    ret_ary = np.zeros([len(array)])
    m       = array[0]
    sum_val = 0

    for i in range(0, size):
        if(m<array[i]):
            m = array[i]

    for i in range(0, size):
        sum_val += np.exp(array[i] - m)

    constant = m + np.log(sum_val)
    for i in range(0, size):
        ret_ary[i] = np.exp(array[i] - constant)
        
    return ret_ary


''' Function to transform a label saved as a char to an hot-one encoded array where the 1 is put in the correct label position '''
def DigitToSoftmax(current_label, known_labels):
    ret_ary = np.zeros(len(known_labels))

    # known_labels_2 = [0,1,2,3,4,5]
                       
    for i in range(0, len(known_labels)):
        if(current_label == known_labels[i]):
            ret_ary[i] = 1

    return ret_ary  



#def NumberToSoftmax(current_label, known_labels):
#    ret_ary = np.zeros(len(known_labels))
#                     
#    for i in range(0, len(known_labels)):
#        if(current_label == known_labels[i]):
#            ret_ary[i] = 1
#
#    return ret_ary

''' Function that initializes a KMean clustering object and trains it on the dataset provided'''
def create_k_mean(data, number_of_clusters, verbose = False):

    # n_jobs is set to -1 to use all available CPU cores. This makes a big difference on an 8-core CPU
    # especially when the data size gets much bigger. #perfMatters

    k = KMeans(n_clusters=number_of_clusters, n_init=100)
    # k = KMeans(n_clusters=number_of_clusters, n_init=20, max_iter=500)

    # Let's do some timings to see how long it takes to train.
    start = time.time()

    # Train it up
    k.fit(data)

    # Stop the timing
    end = time.time()

    # And see how long that took
    if verbose:
        print("Training took {} seconds".format(end - start))

    return k


'''Function to compute kmean clustering on the new dataset and the saved features'''
def k_mean_clustering(features_run, features_saved, labels_run, labels_saved, n_cluster, batch_size):

  # Define initial set of features
  labels_init_list = list(range(0, n_cluster))

  # labels_init_list = list([1, 9, 5, 0])
  # labels_init_list = list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  # n_cluster = len(labels_init_list)

  # Extract from the saved features the labels that we need
  features_saved_init = []
  labels_features_saved_init = []
  # Extract features of digits considered in labels_init_list
  for i in range(0, len(features_saved)):
      if labels_saved[i] in labels_init_list:
        features_saved_init.append(features_saved[i,:])
        labels_features_saved_init.append(labels_saved[i])
  
  # Convert list to nparray
  features = np.array(features_saved_init)
  features = features.astype('float32')
  labels_features = np.array(labels_features_saved_init)  

  # Concateno al vettore delle features iniziali le features della nuova batch da analizzare
  features = np.concatenate((features, features_run))
  labels_features = np.append(labels_features, labels_run).astype(int)

  # Repeat until clustering is correct
  while True:
    # KMean Clustering
    k_mean = create_k_mean(features, n_cluster)

    # Find pseudolabels for each new image
    # Pseudolabels are computed by looking at the confusion matrix of the saved dataset (where ground truth is known)
    clusters_features_saved = list(k_mean.labels_[0:len(labels_features_saved_init)])
    labels_features_saved_init = list(labels_features_saved_init)
    cluster_list = list(range(0,n_cluster))
    map_clu2lbl, map_lbl2clu = cluster_to_label(clusters_features_saved, labels_features_saved_init, cluster_list, labels_init_list) 
 
    if len(map_clu2lbl) == n_cluster:
        # Exit the loop
        break

  clusters_features = k_mean.labels_

  # Compute pseudolabels
  pseudolabels = []
  for i in range(0, len(clusters_features)):
    pseudolabel = map_clu2lbl[clusters_features[i]]
    pseudolabels.append(pseudolabel)


  pseudolabels_run = pseudolabels[len(clusters_features) - batch_size: len(clusters_features)]

  err = 0 # Initialize error counter
  for i in range(len(labels_run)):
    if pseudolabels_run[i] != labels_run[i]:
      err += 1
  
  # Evaluation metrics
  ComputeClusteringMetrics(features, pseudolabels, k_mean)
  ComputeEvalMetrics(labels_run, pseudolabels_run, labels_init_list)

  return pseudolabels_run, err


def trainOneEpoch_OL(model, images, labels, features_saved, labels_saved, batch_size):
       
    learn_rate = model.l_rate
    n_cluster = 10
    n_samples = images.shape[0]

    # BATCH PROCESSING OF DATA
    n_batch = int(np.ceil(n_samples / batch_size))
    images_batch = np.array_split(images, n_batch)
    labels_batch = np.array_split(labels, n_batch)

    err_tot = 0
    err_batch = np.zeros((n_batch,2))
    pseudo_labels = []
    for i in range(0, n_batch):
        print("Starting batch: {}/{}".format(i+1, n_batch))
        # Features extraction
        start1 = time.time()
        features_batch = model.ML_frozen.predict(images_batch[i].reshape((batch_size,28,28,1)), verbose = False)
        end1 = time.time()

        # Kmean clustering
        start2 = time.time()
        pseudo_labels_batch, err = k_mean_clustering(features_batch, features_saved, labels_batch[i], labels_saved, n_cluster, batch_size)
        end2 = time.time()
        pseudo_labels.extend(pseudo_labels_batch)
        err_batch[i, :] = err, len(labels_batch)
        err_tot += err

        print("Features extraction took {:.3f} seconds and Kmean clustering took {:.3f} seconds, with {:.1%} accuracy ({} errors)".format(end1-start1, end2-start2, 1-err/batch_size, err))

    print("Total clustering error: {:.1%} ({}/{} errors)".format(err_tot/n_samples, err_tot, n_samples))

    # ONLINE-LEARNING -> si può spostare nella parte delle batch
    print('**********************************\n Performing training with OL\n')
    features_images = model.ML_frozen.predict(images.reshape((n_samples,28,28,1)), verbose = False)
    for i in range(0, n_samples):
        update_active_layer(model, features_images[i,:], pseudo_labels[i])


'''Function to compute confusion matrix between cluster and labels'''
def confusion_matrix2(clusters_features_saved, labels_features_saved_init, cluster_list, labels_list):

  cmtx = np.zeros([len(labels_list), len(cluster_list)])

  for i in range(0, len(clusters_features_saved)):

    cluster = clusters_features_saved[i]
    label = labels_features_saved_init[i]

    # Find indices
    m = labels_list.index(label)
    n = cluster_list.index(cluster)
    cmtx[m,n] += 1
  return cmtx


'''Function to map the cluster index with the pseudo labels. The function must be run only on the saved_dataset'''
def cluster_to_label(clusters_features, labels_features, cluster_list, labels_init_list):

  # 1: Compute Confusion matrix (for the saved features)
  cmtx = confusion_matrix2(clusters_features, labels_features, cluster_list, labels_init_list)

  # 2: Find max in each row -> cluster corresponding to each label
  map_idx = np.argmax(cmtx, axis = 1)  

  # Fill dictionary with map
  map_clu2lbl = {}
  map_lbl2clu = {}
  labels_init_list.sort()
  for i in range(0, len(map_idx)):
    map_clu2lbl[map_idx[i]] = labels_init_list[i]
    map_lbl2clu[labels_init_list[i]] = map_idx[i]
  
  # Mapping dictionary
  # map_clu2lbl -> cluster: label
  # map_lbl2clu -> label: cluster

  return map_clu2lbl, map_lbl2clu


''' Function to check if the current label is already known to the model (OL layer). If not it augments the custom layer adding a new node'''
def CheckLabelKnown(model, current_label):
    
    found = False
    for i in range(0, len(model.label)):
        if(current_label == model.label[i]):
            found = True
        
    if not found:
        print(f'New digit detected ->', current_label)

        model.label.append(current_label)   # Add new digit to label
                
        # Increase weights and biases dimensions
        model.W = np.hstack((model.W, np.zeros([model.W.shape[0],1])))
        model.b = np.hstack((model.b, np.zeros([1])))
        
        model.W_2 = np.hstack((model.W_2, np.zeros([model.W.shape[0],1])))
        model.b_2 = np.hstack((model.b_2, np.zeros([1])))

def update_active_layer(model, features, pseudolabel):

    learn_rate = model.l_rate

    CheckLabelKnown(model, pseudolabel)
    
    y_true_soft = DigitToSoftmax(pseudolabel, model.label)
               
    # Prediction
    y_pred = model.predict(features)
        
    # Backpropagation
    cost = y_pred-y_true_soft
        
    for j in range(0,model.W.shape[0]):

         # Update weights
        dW = np.multiply(cost, features[j]*learn_rate)
        model.W[j,:] = model.W[j,:]-dW

    # Update biases
    db      = np.multiply(cost, learn_rate)
    model.b = model.b-db



class Custom_Layer(object):
    def __init__(self, model):

        # Related to the layer
        self.ML_frozen = keras.models.Sequential(model.layers[:-1])  # extract the last layer from the original model
        self.ML_frozen.compile()
        
        self.W = np.array(model.layers[-1].get_weights()[0])    # extract the weights from the last layer
        self.b = np.array(model.layers[-1].get_weights()[1])    # extract the biases from the last layer
               
        self.W_2 = np.zeros(self.W.shape)
        self.b_2 = np.zeros(self.b.shape)
        
        self.label     = [0,1,2,3,4,5]              
        self.std_label = [0,1,2,3,4,5,6,7,8,9]
        
        self.l_rate = 0                                         # learning rate that changes depending on the algorithm        

        self.batch_size = 0
        
        # Related to the results fo the model
        self.conf_matr = np.zeros((10,10))    # container for the confusion matrix       
        self.macro_avrg_precision = 0       
        self.macro_avrg_recall = 0
        self.macro_avrg_F1score = 0
        
        self.title = ''       # title that will be displayed on plots
        self.filename = ''    # name of the files to be saved (plots, charts, conf matrix)
        
    # Function that is used for the prediction of the model saved in this class
    def predict(self, x):
        mat_prod = np.array(np.matmul(x, self.W) + self.b)
        return softmax(mat_prod) # othwerwise do it with keras|also remove np.array()| tf.nn.softmax(mat_prod) 