import numpy as np
import cv2
import serial.tools.list_ports
import serial, struct
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")

import time
from sklearn.cluster import KMeans

from numpy.ma.core import size

import sys, os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_PATH + '/lib')

from Kmeans_lib import *
from EvalMetrics import *
from simulation_lib import *

from importMnist import createDataset

# --------------------------------------------------------------------------
#    _______  ______  _        _    _   _    _  _____ ___ ___  _   _
#   | ____\ \/ /  _ \| |      / \  | \ | |  / \|_   _|_ _/ _ \| \ | |
#   |  _|  \  /| |_) | |     / _ \ |  \| | / _ \ | |  | | | | |  \| |
#   | |___ /  \|  __/| |___ / ___ \| |\  |/ ___ \| |  | | |_| | |\  |
#   |_____/_/\_\_|   |_____/_/   \_\_| \_/_/   \_\_| |___\___/|_| \_|

"""
This python script is used for sincronizing the OpenMV camera and the laptop during training. The idea is to disaply digits images on the 
laptop screen and at the same time send throught the UART (usb cable) to the OpenMV camera the correct label of the image displayed.
This should allow the camera to have the true label and correctly compute the error and later perform the backpropagation on biases and weights 
in order to perform the OL training.

Note that the UART on the USB cable is usually occupied by the OpenMV IDE, which will use this cable for receiving all the debugging informations from
the OpenMV camera (such as the video stream). In order to be able to communicate the informations from the Laptop to the OpenMV camera it is necessary to
flash the MicroPython code on the camera as the main.py script (in the IDE go to Tools->Save opened scipt as main.py). In this way, any time the camera
is powered on and NOT connected in debugging mode to the IDE (in the IDE in the bottom left corner you should see the disconnected image), the main.py script
is ran automatically and is possible to use the UART connection for sending and receiveing data (also the photos taken from the camera can be sent to the 
PC but for sure the code will be slower and it can easily get out of sync).
"""


############################################################
#    _____ _   _ _   _  ____ _____ ___ ___  _   _ ____
#   |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___|
#   | |_  | | | |  \| | |     | |  | | | | |  \| \___ \
#   |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/


##################################################################
# DEFINITION OF THE FUNCTIONS USED FOR THE OPENMV
##################################################################

def on_change(val):
    # Function called when the sliding bar changes value.                                            #
    # The function changes the value TRAINING FLAG that is used as a indicator of the state machine. #
    # Parameters                                                                                     #
    # ..........                                                                                     #
    # val : integer                                                                                  #
    # Is the value set by the sliding bar                                                            #

    myClass.TRAINING_FLAG = val
    if (val == 0):
        print('Script is in IDLE MODE')
    elif (val == 1):
        print('Script is in STREAMING MODE')
    elif (val == 2):
        print('Script is in STREAMING ELABORATION MODE')
    elif (val == 3):
        print('Script is in TRAINING MODE')



class uselessContainer():
    # Container that I use because I need to change the parameter TRAINING_FLAG            #
    # if I don't use a class the value is not changed by ID and the script never           #
    # updates the real value but creates a new value with the same name but different ID   #
    
    def __init__(self):
        self.TRAINING_FLAG = 3
        self.cont = 0 

##################################################################
# DEFINITION OF THE FUNCTIONS USED IN THE ONLINE LEARNING
##################################################################

def trainOneEpoch_CWR(model, x_test, y_test, features, labels_features, clust_batch_size, found_digit):
    
    learn_rate = model.l_rate
    model_batch_size = model.batch_size

    test_samples = x_test.shape[0]
    
    n_cluster = 10
    
    err_clu = 0
    err_mod = 0
    
    prediction_vec = np.zeros(test_samples)

    # CLUSTERING
    print('**********************************\n Performing clustering\n')
    
    # Pseudo-labels
    pseudo_labels, err_clu = k_mean_clustering(x_test, features, y_test, labels_features, n_cluster, clust_batch_size)

    # ONLINE-LEARNING
    print('**********************************\nPerforming training CWR \n ')  
                   
    # Cycle over all input samples
    for i in range(0, test_samples):
            
        CheckLabelKnown(model, pseudo_labels[i])
        y_true_soft = NumberToSoftmax(pseudo_labels[i], model.label)
        
        h = model.W.shape[0]
        w = model.W.shape[1] 

        found_digit[np.argmax(y_true_soft)] += 1  # update the digit counter
            
        # PREDICTION

        y_pred_c = softmax(np.array(np.matmul(x_test[i,:], model.W) + model.b))      
        y_pred_t = softmax(np.array(np.matmul(x_test[i,:], model.W_2) + model.b_2)) 
        
        prediction_vec[i] = np.argmax(y_pred_c)

        # Error
        if(prediction_vec[i] !=  y_test[i]):
        #if(np.argmax(y_pred) !=  y_test):  
            err_mod += 1

        # BACKPROPAGATION
        cost = y_pred_t-y_true_soft

        # Update weights
        for j in range(0,h):
            deltaW = np.multiply(cost, x_test[i,j])
            dW = np.multiply(deltaW, learn_rate)
            model.W_2[j,:] = model.W_2[j,:] - dW

        # Update biases
        db = np.multiply(cost, learn_rate)
        model.b_2 = model.b_2-db
        
        
        # If beginning of batch
        if(i%model_batch_size==0 and i!=0): 
            for k in range(0, w):
                if(found_digit[k]!=0):
                    tempW = np.multiply(model.W[:,k], found_digit[k])
                    tempB = np.multiply(model.b[k]  , found_digit[k])
                    model.W[:,k] = np.multiply(tempW+model.W_2[:,k], 1/(found_digit[k]+1))
                    model.b[k]   = np.multiply(tempB+model.b_2[k],   1/(found_digit[k]+1))
                    
            model.W_2  =  np.copy(model.W) # np.zeros((model.W.shape)) 
            model.b_2  =  np.copy(model.b) # np.zeros((model.b.shape))       
            found_digit = np.zeros(10)  # reset
        
    # Metrics

    print('\n*******************************************************************************')
    print('***** Model batch accuracies: \n')
    
    ComputeEvalMetrics(y_test, prediction_vec, list(range(0, n_cluster)))

    
    return np.array(pseudo_labels).astype(int), prediction_vec, err_clu, err_mod

def trainOneEpoch_OL(model, x_test, y_test, features, labels_features, batch_size):
    
    learn_rate = model.l_rate

    test_samples = x_test.shape[0]
    
    n_cluster = 10
    
    err_clu = 0
    err_mod = 0
    
    prediction_vec = np.zeros(test_samples)
      
    
    # CLUSTERING
    print('**********************************\n Performing clustering\n')
    
    # Pseudo-labels

    pseudo_labels, err_clu = k_mean_clustering(x_test, features, y_test, labels_features, n_cluster, batch_size)
    
    # ONLINE-LEARNING
    print('**********************************\n Performing training with OL\n')

    for i in range(0, test_samples):

        CheckLabelKnown(model, pseudo_labels[i])
    
        y_true_soft = NumberToSoftmax(pseudo_labels[i], model.label)
               
        # Prediction
        y_pred = model.predict(x_test[i,:])
        prediction_vec[i] = np.argmax(y_pred)
        
        # Error
        if(prediction_vec[i] !=  y_test[i]):
        #if(np.argmax(y_pred) !=  y_test):
            err_mod += 1
        
        # Backpropagation
        cost = y_pred-y_true_soft
        
        for j in range(0,model.W.shape[0]):

            # Update weights
            dW = np.multiply(cost, x_test[i,j]*learn_rate)
            model.W[j,:] = model.W[j,:]-dW

        # Update biases
        db      = np.multiply(cost, learn_rate)
        model.b = model.b-db

    
    #y_true_soft = NumberToSoftmax(y_test, model.label)
                   
    # Find the max iter for both true label and prediction
    #if(np.amax(y_true_soft) != 0):
    #    max_i_true = np.argmax(y_true_soft)

    #if(np.amax(y_pred) != 0):
    #    max_i_pred = np.argmax(y_pred)

    # Fill up the confusion matrix
    #for k in range(0,len(model.label)):
    #    if(model.label[max_i_pred] == model.std_label[k]):
    #        p = np.copy(k)
    #    if(model.label[max_i_true] == model.std_label[k]):
    #        t = np.copy(k)

    #model.conf_matr[t,p] += 1 
    
    return pseudo_labels, prediction_vec, err_clu, err_mod

###################################
#    __  __    _    ___ _   _
#   |  \/  |  / \  |_ _| \ | |
#   | |\/| | / _ \  | ||  \| |
#   | |  | |/ ___ \ | || |\  |
#   |_|  |_/_/   \_\___|_| \_|

#################################################
# PARAMETERS FOR IMPLEMENTATION OF ACTIVE MODEL
#################################################

# Path of the images to open
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

myClass = uselessContainer()  # Init the class that stores the state of the camera

# Open serial port
# Next lines are taken from the example script in the OpenMV IDE - the example is in    File->Examples->OpenMV->Board Control->usb_vcp.py
# NB: see the name of the com port used from the camera in                              Windows->Device manager->Ports(COM and LPT)
port = '/dev/tty.usbmodem3067376B30301'
sp = serial.Serial(port, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, xonxoff=False,
                rtscts=False, stopbits=serial.STOPBITS_ONE, timeout=5000, dsrdtr=True)
sp.setDTR(True)

# Import the dataset that I am going to display
samples_for_each_digit = 100
digits_i_want = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

digits_data, digits_label = createDataset(samples_for_each_digit, digits_i_want)  # load dataset ; originally was createDataset(samples_for_each_digit + 1, digits_i_want)
tot_samples = len(digits_label)                                                   # original is len(digits_label) - 8

testing_number = 1000
training_number = tot_samples - testing_number

#########################################################
# EXTRA PARAMETERS FOR IMPLEMENTATION OF ACTIVE MODEL
#########################################################

ref_feat = np.loadtxt('ll_feat_10_70%.txt')

labels_features = np.loadtxt('ll_lab_feat_10_70%.txt')

model = keras.models.load_model('mnist_cnn.h5')

out_collect = []

batch_size = 50
err_cluster = 0
err_model = 0
cntr = 1
cntr_batch = 1

found_digit = np.zeros(10)


Model_OL = Custom_Layer(model)
Model_OL.title      = 'OL'
Model_OL.filename   = 'OL'
Model_OL.W_2  = np.zeros((Model_OL.W.shape))
Model_OL.b_2  = np.zeros((Model_OL.b.shape))
Model_OL.l_rate     = 0.0001
Model_OL.batch_size = 10


######################
# START OF THE CODE
######################

print('\n\n ***** EVERYTHING IS LOADED - READY TO RUN ***** \n\n')

while 1:

    # Show digit/idle message
    if (myClass.TRAINING_FLAG != 0):
        zoom_digit = cv2.resize(digits_data[cntr-1], (0, 0), fx=7, fy=7)
        cv2.imshow('SYNC APP', zoom_digit)
        cv2.waitKey(150)


    # Send cmd + label to OpenMV
    if (myClass.TRAINING_FLAG == 0 or myClass.TRAINING_FLAG == 1):
        sp.write(b"snap")  # the camera will be in streaming mode
        sp.flush()

        # Receive image from OpenMV - careful it's easy to get out of sync
        size = struct.unpack('<L', sp.read(4))[0]
        img_raw = sp.read(size)
        img_openmv = cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_COLOR)
        zoom_openmv = cv2.resize(img_openmv, (0, 0), fx=5, fy=5)
        cv2.imshow('OpenMV view - Zoomed', zoom_openmv)
        cv2.waitKey(150)


    elif (myClass.TRAINING_FLAG == 2):
        sp.write(b"elab")  # the camera will be in streaming mode
        sp.flush()
        
        # Receive image from OpenMV - careful it's easy to get out of sync
        size = struct.unpack('<L', sp.read(4))[0]
        img_raw = sp.read(size)
        img_openmv = cv2.imdecode(np.frombuffer(img_raw, np.uint8), 0)
        zoom_openmv = cv2.resize(img_openmv, (400, 400))
        cv2.imshow('OpenMV view - Zoomed', zoom_openmv)
        cv2.waitKey(150)

        
    elif (myClass.TRAINING_FLAG == 3):

        if cntr == 1:
            cv2.waitKey(8000)
        
        sp.write(b"trai")  # the camera will train on the image taken
        sp.flush()

        print(f'counter: ',cntr,'/',tot_samples,'\n')

        feature = sp.read(3072).decode("utf-8")
        sp.flush()

        # This section is a post processing in case of firmware.bin implementation on OpenMV.      #
        # Sometimes the converted features are bigger than 512 and I have to remove elements.      #
        # I choose void strings and strings containing only integer elements inside to be removed  #
        
        feature = feature[0:len(feature)-2].split(',')
        feature_new = (np.asarray([float(i) for i in feature])).reshape(1,len(feature))

        # Generation of the batch vector                                                     
        if cntr_batch == 1:
            feature_batch = np.copy(feature_new)
            label_batch= np.copy(digits_label[cntr-1])
        if cntr_batch > 1:
            feature_batch= np.concatenate((feature_batch,feature_new))
            label_batch = np.append(label_batch,digits_label[cntr-1])

        # Elaboration
        if(cntr_batch == batch_size):
            
            # Make labels integers and rounds to 3 decimals the features
            label_batch = label_batch.astype(int)
            labels_features = labels_features.astype(int)
            
            #pseudo_labels, predictions, err_clu, err_mod = trainOneEpoch_OL(Model_OL, feature_batch, label_batch, ref_feat, labels_features, batch_size)
            pseudo_labels, predictions, err_clu, err_mod = trainOneEpoch_CWR(Model_OL, feature_batch, label_batch, ref_feat, labels_features, batch_size, found_digit)

            # Error calculation
            if cntr > training_number:
                err_cluster = err_cluster + err_clu
                err_model = err_model + err_mod

            if cntr == cntr_batch:
                pseudo_labels_tot = np.copy(pseudo_labels)
                predictions_tot= np.copy(predictions)
            if cntr > cntr_batch:
                pseudo_labels_tot= np.append(pseudo_labels_tot,pseudo_labels)
                predictions_tot = np.append(predictions_tot,predictions)

            # Counter reset
            cntr_batch = 0

            # Debug
            #print('\n*******************************************************************************')
            #print('***** Currently at ', int(cntr/tot_samples*100), '%')
            #print('***** Pseudo_labels: ', pseudo_labels.astype(int))
            #print('***** Predictions:   ', predictions.astype(int))
            #print('***** True labels:   ', label_batch)
            #print('\n*******************************************************************************\n')      

        # Counters Update
        cntr_batch +=1
        cntr += 1 

    # Condition for exiting the loop at end training
    if (cntr == tot_samples + 1):
        sp.write(b"endt")
        myClass.TRAINING_FLAG = 5
        myClass.cont += 1
    
    if (myClass.cont == 1):
        break

print('\n*******************************************************************************')
print('***** Clustering accuracies: \n')
ComputeEvalMetrics(digits_label, pseudo_labels_tot, list(range(0, 10)))

print('\n*******************************************************************************')
print('***** Model accuracies: \n')
ComputeEvalMetrics(digits_label, predictions_tot, list(range(0, 10)))

# Debug
#print('\n*******************************************************************************')
#print('***** Clustering accuracy: ', int((testing_number-err_cluster)/testing_number*100),'%')
#print('\n*******************************************************************************\n')

#print('\n*******************************************************************************')
#print('***** Model accuracy:      ', int((testing_number-err_model)/testing_number*100),'%')
#print('\n*******************************************************************************\n')

print('*******************************************************************************')
print('***** The training images are finished, press ANY KEY to close the script *****')
print('*******************************************************************************')