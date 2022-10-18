# import numpy as np
from lib.Kmeans_lib import confusion_matrix

# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def ComputeEvalMetrics(true_label, pred_label, cluster_list, labels_list):

    # IMPORTARE CONFUSION MATRIX CUSTOM-MADE 








    # confusion = confusion_matrix(true_label, pred_label, cluster_list, labels_list)
    # print('Confusion Matrix\n')
    # print(confusion)




    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(true_label, pred_label)))

    print('Micro Precision: {:.2f}'.format(precision_score(true_label, pred_label, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(true_label, pred_label, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(true_label, pred_label, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(true_label, pred_label, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(true_label, pred_label, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(true_label, pred_label, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(true_label, pred_label, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(true_label, pred_label, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(true_label, pred_label, average='weighted')))

    print('\nClassification Report\n')
    print('TO DO')
    # print(classification_report(true_label, pred_label, target_names=['A', 'E', 'I', 'O', 'U']))

