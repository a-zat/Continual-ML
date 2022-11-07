from sklearn.cluster import KMeans
import time
import numpy as np

from lib.EvalMetrics import *
from lib.Kmeans_lib import cluster_to_label

'''Function to compute kmean clustering on the new dataset and the saved features'''
def create_kmean(features_saved, labels_saved, model):

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

  return map_clu2lbl, kmeans


'''Function to compute kmean clustering on the new dataset and the saved features'''
def update_kmean(features_run, labels_run, kmeans, map_clu2lbl, model):

  # Passo una nuova immagine al Kmeans. Ne determino il cluster e ne calcolo la pseudolabel
  errs = 0
  n_samples = len(labels_run)
  cluster_label = np.zeros(n_samples, dtype=int)
  pseudolabels = []
  
  for i in range(0, n_samples):
      # Find the cluster for the new features
      cluster_label[i] = kmeans.predict(features_run[i,:].reshape(1, -1))
      pseudolabel = map_clu2lbl[cluster_label[i]]
      pseudolabels.append(pseudolabel)

      # Update the cluster center
      l_rate = 0.02
      kmeans.cluster_centers_[cluster_label[i],:] = (kmeans.cluster_centers_[cluster_label[i],:] + features_run[i,:] * l_rate)/(1 + l_rate)
      # print(cluster_label[i])

      if labels_run[i] != pseudolabel:
          errs += 1
  
  # Evaluation metrics
  if model.settings.verbosity == 'DEBUG':
    ComputeClusteringMetrics(features_run, pseudolabels, kmeans)
    # ComputeEvalMetrics(labels_run, pseudolabels, labels_init_list)


  # print("Errors:", errs, "Accuracy: {:.1%}".format(1- errs/n_samples))

  return pseudolabels, errs


from lib.CustomLayer_lib import update_ll
from lib.utils import UpdateConfusion

'''Script to run one epoch'''
def RunOneEpoch_V2(model, images, labels, features_saved, labels_saved):

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
    predictions = []

    if model.ll_method == 'CWR':
        model_cntr = 0 
        found_digit = np.zeros(10) 

    # Create kmeans
    map_clu2lbl, kmeans = create_kmean(features_saved, labels_saved, model)
    
    # Update
    for i in range(0, n_batch):
        batch_size = labels_batch[i].shape[0]# Current batch size
        print("Starting {} batch: {}/{}".format(model.settings.mode, i+1, n_batch))
        # Features extraction
        start1 = time.time()
        features_batch = model.ML_frozen.predict(images_batch[i].reshape((batch_size,28,28,1)), verbose = False)
        features_batch = np.array(features_batch, dtype = type(features_saved[0,0]))
        end1 = time.time()

        # Kmean clustering
        start2 = time.time()
        pseudolabels_batch, err_clu_batch = update_kmean(features_batch, labels_batch[i], kmeans, map_clu2lbl, model)
        end2 = time.time()
        pseudolabels.extend(pseudolabels_batch)
        err_clu += err_clu_batch
        clust_err_array.append(err_clu_batch)

        # Last Layer update
        err_mod_batch = 0
        for j in range(batch_size):

            prediction = update_ll(model, features_batch[j,:], pseudolabels_batch[j])
            predictions.append(prediction)

            if(prediction != labels_batch[i][j]):  
               err_mod_batch += 1

        # Update confusion matrix
            if model.settings.fill_cmtx == True:
                UpdateConfusion(model, prediction, labels_batch[i][j], 'model')
                UpdateConfusion(model, pseudolabels_batch[j], labels_batch[i][j], 'clust')

        model_err_array.append(err_mod_batch)
        err_mod += err_mod_batch

        if model.settings.verbosity == 'EOBINFO':
            print("Features extraction took {:.3f} seconds and Kmean clustering took {:.3f} seconds, with {:.1%} accuracy ({} errors)".format(end1-start1, end2-start2, 1-err_clu_batch/batch_size, err_clu_batch))
            print("Batch Model errors {} ({:.1%} accuracy)".format(err_mod_batch, 1-err_mod_batch/batch_size))
    
    if model.settings.verbosity == 'EOEINFO' or model.settings.verbosity == 'EOBINFO':
        print("Total clustering error: {:.1%} ({}/{} errors, {:.1%} accuracy)".format(err_clu/n_samples, err_clu, n_samples, 1-err_clu/n_samples))
        print("Total model error: {:.1%} ({}/{} errors, {:.1%} accuracy)".format(err_mod/n_samples, err_mod, n_samples, 1-err_mod/n_samples))

    if model.settings.save_output == True:
        model.settings.datalog = list([err_clu, err_mod, clust_err_array, model_err_array])

    if model.settings.save_extralog == True:
        model.settings.extralog = list([labels, pseudolabels, predictions])