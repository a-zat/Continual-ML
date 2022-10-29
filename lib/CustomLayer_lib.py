# Library for custom layer
import numpy as np
import time
from tensorflow import keras

from lib.Kmeans_lib import k_mean_clustering

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
def LabelToActivation(current_label, known_labels):
    ret_ary = np.zeros(len(known_labels))

    # known_labels_2 = [0,1,2,3,4,5]
                       
    for i in range(0, len(known_labels)):
        if(current_label == known_labels[i]):
            ret_ary[i] = 1

    return ret_ary  

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


''' Custom layer class'''
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

        self.ll_method = ''
        
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

'''Class to store settings for training/testing the model'''
class TrainSettings(object):
    def __init__(self):
        self.cluster_batch_size = 1
        self.verbosity = 'SILENT'
        self.fill_cmtx = True
        self.clustering_labels = []
        self.n_cluster = len(self.clustering_labels)
        self.save_output = False
        self.save_path = ''
        self.save_plots = False
        self.mode = 'UNDEFINED'

def update_ll_OL(model, features, pseudolabel):
    learn_rate = model.l_rate
    
    CheckLabelKnown(model, pseudolabel)
    
    y_true_soft = LabelToActivation(pseudolabel, model.label)
               
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

    prediction = model.label[np.argmax(y_pred)]

    return prediction



def update_ll_CWR(model, features, pseudolabel, found_digit, reset):
    learn_rate = model.l_rate
    model_batch_size = model.batch_size
           
    CheckLabelKnown(model, pseudolabel)
    y_true_soft = LabelToActivation(pseudolabel, model.label)
        
    h = model.W.shape[0]
    w = model.W.shape[1] 

    found_digit[np.argmax(y_true_soft)] += 1  # update the digit counter
            
    # PREDICTION
    y_pred_c = softmax(np.array(np.matmul(features, model.W) + model.b))      
    y_pred_t = softmax(np.array(np.matmul(features, model.W_2) + model.b_2)) 
 
    # BACKPROPAGATION
    cost = y_pred_t-y_true_soft

    # Update weights
    for j in range(0,h):
        dW = np.multiply(cost, features[j] * learn_rate)
        model.W_2[j,:] = model.W_2[j,:] - dW

    # Update biases
    db = np.multiply(cost, learn_rate)
    model.b_2 = model.b_2-db     
        
    #If beginning of batch
    if(reset):
        for k in range(0, w):
            if(found_digit[k]!=0):
                tempW = np.multiply(model.W[:,k], found_digit[k])
                tempB = np.multiply(model.b[k]  , found_digit[k])
                model.W[:,k] = np.multiply(tempW+model.W_2[:,k], 1/(found_digit[k]+1))
                model.b[k]   = np.multiply(tempB+model.b_2[k],   1/(found_digit[k]+1))
        model.W_2  =  np.copy(model.W) # np.zeros((model.W.shape)) 
        model.b_2  =  np.copy(model.b) # np.zeros((model.b.shape))       
        found_digit = np.zeros(10)  # reset
    

    prediction = model.label[np.argmax(y_pred_c)]
    return prediction, found_digit

    


