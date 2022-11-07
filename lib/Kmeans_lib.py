from sklearn.cluster import KMeans
import time
import numpy as np

from lib.EvalMetrics import *

''' Function that initializes a KMean clustering object and trains it on the dataset provided'''
def create_k_mean(data, n_clusters, verbose = False):

    # n_jobs is set to -1 to use all available CPU cores. This makes a big difference on an 8-core CPU
    # especially when the data size gets much bigger.

    k = KMeans(n_clusters, n_init=100)

    # Let's do some timings to see how long it takes to train.
    start = time.time()

    # Train it up
    k.fit(data)

    # Stop the timing
    end = time.time()

    # And see how long that took
    if verbose:
        print("Fitting {} samples in {} clusters took {} seconds".format(data.shape[0], n_clusters, end - start))

    return k


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
def cluster_to_label(clusters_features, labels_features, cluster_list, labels_init_list, verbose = False):

  # 1: Compute Confusion matrix (for the saved features)
  cmtx = confusion_matrix2(clusters_features, labels_features, cluster_list, labels_init_list)

  # 2: Find max in each row -> cluster corresponding to each label
  map_idx = np.argmax(cmtx, axis = 0)  

  # Fill dictionary with map
  map_clu2lbl = {}
  map_lbl2clu = {}
  labels_init_list.sort()
  for i in range(0, len(map_idx)):
    map_lbl2clu[map_idx[i]] = labels_init_list[i]
    map_clu2lbl[labels_init_list[i]] = map_idx[i]
  
  # Mapping dictionary
  # map_clu2lbl -> cluster: label
  # map_lbl2clu -> label: cluster

  if verbose:
    print(cmtx)
    print("Argmax:", map_idx)
    print("Cluster to label map: ", map_clu2lbl)

  return map_clu2lbl, map_lbl2clu


'''Function to compute kmean clustering on the new dataset and the saved features'''
def k_mean_clustering(features_run, features_saved, labels_run, labels_saved, model):

  # Define initial set of features
  labels_init_list = model.std_label
  n_cluster = len(labels_init_list)

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
  max_iter = 5
  iter = 0
  while True:
    # KMean Clustering
    k_mean = create_k_mean(features, n_cluster, verbose = (model.settings.verbosity == 'DEBUG'))

    # Find pseudolabels for each new image
    # Pseudolabels are computed by looking at the confusion matrix of the saved dataset (where ground truth is known)
    clusters_features_saved = list(k_mean.labels_[0:len(labels_features_saved_init)])
    labels_features_saved_init = list(labels_features_saved_init)
    cluster_list = list(range(0,n_cluster))
    map_clu2lbl, map_lbl2clu = cluster_to_label(clusters_features_saved, labels_features_saved_init, cluster_list, labels_init_list, verbose = (model.settings.verbosity == 'DEBUG'))
 
    iter += 1
    if len(map_clu2lbl) == n_cluster or iter > max_iter:
        # Exit the loop
        break

  clusters_features = k_mean.labels_

  # Compute pseudolabels
  pseudolabels = []
  for i in range(0, len(clusters_features)):
    pseudolabel = map_clu2lbl[clusters_features[i]]
    pseudolabels.append(pseudolabel)

  pseudolabels_run = pseudolabels[len(clusters_features) - len(labels_run): len(clusters_features)]

  err = 0 # Initialize error counter
  for i in range(len(labels_run)):
    if pseudolabels_run[i] != labels_run[i]:
      err += 1
  
  # Evaluation metrics
  if model.settings.verbosity == 'DEBUG':
    ComputeClusteringMetrics(features, pseudolabels, k_mean)
    ComputeEvalMetrics(labels_run, pseudolabels_run, labels_init_list)

  return pseudolabels_run, err



'''Function to compute kmean clustering on the new dataset and the saved features'''
def k_mean_clustering2(features_run, features_saved, labels_run, labels_saved, model):

  # Define initial set of features
  labels_init_list = model.std_label
  n_cluster = len(labels_init_list)

  # Extract from the saved features the labels that we need
  features_saved_init = []
  labels_saved_init = []
  # Extract features of digits considered in labels_init_list
  for i in range(0, len(features_saved)):
      if labels_saved[i] in labels_init_list:
        features_saved_init.append(features_saved[i,:])
        labels_saved_init.append(labels_saved[i])
  
  # Convert list to nparray
  features = np.array(features_saved_init)
  features = features.astype('float32')
  labels_features = np.array(labels_saved_init)  

  # Creo un dizionario per linkare le features (salvate) al cluster di appartenenza
  # creates dictionary using dictionary comprehension -> list [] is mutable object
  features_saved_dict = { key : [] for key in labels_init_list}

  for i in range(0, len(features_saved_init)):
    lbl = labels_saved_init[i]
    features_saved_dict[lbl].append(features_saved_init[i])

  # Converto list-of-arrays in 2D array
  cluster_mean = []
  for key in labels_init_list:
    features_saved_dict[key] = np.array(features_saved_dict[key])
    cluster_mean.append(np.mean(features_saved_dict[key], axis=0))

  cluster_mean = np.array(cluster_mean)

  # Create KMeans
  kmeans = KMeans(n_cluster)
  kmeans.fit(cluster_mean)
  map_clu2lbl, map_lbl2clu = cluster_to_label(kmeans.labels_, labels_init_list, list(range(0,n_cluster)), model.std_label)

  # print(kmeans.predict(cluster_mean))
  # print("Map cluster to label:", map_clu2lbl)

  # Passo una nuova immagine al Kmeans. Ne determino il cluster e ne calcolo la pseudolabel
  errs = 0
  n_samples = len(labels_run)
  cluster_label = np.zeros(n_samples, dtype=int)
  pseudolabels = []
  
  for i in range(0, n_samples):
      labels_new = labels_run[i]
      features_new = np.array(features_run[i,:], dtype = type(features_saved[0,0]))

      # Find the cluster for the new features
      cluster_label[i] = kmeans.predict(features_new.reshape(1, -1))
      pseudolabel = map_clu2lbl[cluster_label[i]]
      pseudolabels.append(pseudolabel)

      # Update the cluster center
      l_rate = 0.02
      kmeans.cluster_centers_[cluster_label[i],:] = (kmeans.cluster_centers_[cluster_label[i],:] + features_new * l_rate)/(1 + l_rate)
      # print(cluster_label[i])

      if labels_new != pseudolabel:
          errs += 1
  
  # Evaluation metrics
  if model.settings.verbosity == 'DEBUG':
    ComputeClusteringMetrics(features, pseudolabels, kmeans)
    ComputeEvalMetrics(labels_run, pseudolabels, labels_init_list)


  print("Errors:", errs, "Accuracy: {:.1%}".format(1- errs/n_samples))

  return pseudolabels, errs
