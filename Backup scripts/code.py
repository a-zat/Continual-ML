#######################################
#  ____   _    ____  ____  _____ 
# |  _ \ / \  |  _ \/ ___|| ____|
# | |_) / _ \ | |_) \___ \|  _|  
# |  __/ ___ \|  _ < ___) | |___ 
# |_| /_/   \_\_| \_\____/|_____|


""" This function shuffles the dataset given as output, the randomization can be changed by chaanging the seed """
def shuffleDataset(data_matrix, lable_ary):
   
    random.seed(56)

    order_list = list(range(0,data_matrix.shape[0]))  
    random.shuffle(order_list)                         

    data_matrix_shuff = np.zeros(data_matrix.shape)
    lable_ary_shuff   = np.empty(data_matrix.shape[0], dtype=str) 

    for i in range(0, data_matrix.shape[0]):
        data_matrix_shuff[i] = data_matrix[order_list[i]] 
        lable_ary_shuff[i]   = lable_ary[order_list[i]]

    return data_matrix_shuff, lable_ary_shuff





#############################################
#  _____ ___ _   ___   __   ___  _     
# |_   _|_ _| \ | \ \ / /  / _ \| |    
#   | |  | ||  \| |\ V /  | | | | |    
#   | |  | || |\  | | |   | |_| | |___ 
#   |_| |___|_| \_| |_|    \___/|_____|



""" The function transforms a label saves as a char in an hot one encoded array where the 1 is put in the correct label space """
def letterToSoftmax(current_label, known_labels):
    ret_ary = np.zeros(len(known_labels))
                       
    for i in range(0, len(known_labels)):
        if(current_label == known_labels[i]):
            ret_ary[i] = 1

    return ret_ary  

""" Function that computes the softmax operator of the array in input.
    Slightly differs from the one implemented by Keras but is needed to maintain consistency here and in the OpenMV camera """
def myFunc_softmax(array):
    
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


import random
import matplotlib.pyplot as plt 
import os
import csv 
import pandas as pd
import re
import random
import matplotlib.image as mpimg
from tensorflow.keras import optimizers
from PIL import Image
import seaborn as sns

