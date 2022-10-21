# import numpy as np
# from lib.Kmeans_lib import confusion_matrix2

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

def ComputeClusteringMetrics(features, labels, k_mean):
    print('Clustering metrics\n')
    print('Silhouette Precision: {:.2f}'.format(silhouette_score(features, labels)))
    print('Calinks-Harabasz Recall: {:.2f}'.format(calinski_harabasz_score(features, labels)))
    print('Davies-Bouldin -score: {:.2f}\n'.format(davies_bouldin_score(features, labels)))
    print('Clusters inertia: {:.2f}\n'.format(k_mean.inertia_))

    # Ci sarebbero anche score(x) e transform(x) in sklearn


def ComputeEvalMetrics(true_labels, pred_labels, labels_list):

    #print("True labels:", true_labels)
    #print("Predicted labels:", pred_labels)

    confusion = confusion_matrix(true_labels, pred_labels, labels = labels_list)
    print('Confusion Matrix\n')
    print(confusion) # Implement fancy confusion matrix -> Confusion Display of sklearn

    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(true_labels, pred_labels)))

    print('Micro Precision: {:.2f}'.format(precision_score(true_labels, pred_labels, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(true_labels, pred_labels, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(true_labels, pred_labels, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(true_labels, pred_labels, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(true_labels, pred_labels, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(true_labels, pred_labels, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(true_labels, pred_labels, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(true_labels, pred_labels, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(true_labels, pred_labels, average='weighted')))

    # print('\nClassification Report\n')
    # print('TO DO')
    # print(classification_report(true_labels, pred_labels, target_names= labels_list))#target_names=['A', 'E', 'I', 'O', 'U']))


    #plot_barChart(Model_KERAS)
    #plot_confMatrix(Model_KERAS)
    #plot_table(Model_KERAS)