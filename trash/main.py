# Includes
from tensorflow.keras.datasets import mnist

# Load dataset
(data_train, label_train),(data_test, label_test) = mnist.load_data() # Load data
print('The original dataset shapes from MNIST are')
print(f'    Train dataset shape: {data_train.shape}')
print(f'    Test dataset shape:  {data_test.shape}')



def update_ll_OL_epoch(model, features, pseudo_labels, true_labels):

    err = 0
    predictions = []
    for i in range(0, len(pseudo_labels)):

        learn_rate = model.l_rate

        CheckLabelKnown(model, pseudo_labels[i])
    
        y_true_soft = LabelToActivation(pseudo_labels[i], model.label)
               
        # Prediction
        y_pred = model.predict(features[i,:])
        
        # Backpropagation
        cost = y_pred-y_true_soft
        
        for j in range(0,model.W.shape[0]):

            # Update weights
            dW = np.multiply(cost, features[i,j]*learn_rate)
            model.W[j,:] = model.W[j,:]-dW

        # Update biases
        db      = np.multiply(cost, learn_rate)
        model.b = model.b-db
        
        # the next part is only to plot the confusion matrix
        # if the train data is finished still train the model but do not save the results
        if(i>=len(pseudo_labels)/2):

            y_true_soft = LabelToActivation(true_labels[i], model.label)
                   
            # Find the max iter for both true label and prediction
            if(np.amax(y_true_soft) != 0):
                max_i_true = np.argmax(y_true_soft)

            if(np.amax(y_pred) != 0):
                max_i_pred = np.argmax(y_pred)

            # Fill up the confusion matrix
            for k in range(0,len(model.label)):
                if(model.label[max_i_pred] == model.std_label[k]):
                    p = np.copy(k)
                if(model.label[max_i_true] == model.std_label[k]):
                    t = np.copy(k)

            model.conf_matr[t,p] += 1  

        y_pred_amax = model.label[np.argmax(y_pred)]
        predictions.append(y_pred_amax)
        if(y_pred_amax != true_labels[i]):
            err += 1

    return predictions, err

def update_ll_CWR_batch(model, features, pseudo_labels, true_labels):

    found_digit = np.zeros(10)

    learn_rate = model.l_rate
    model_batch_size = model.batch_size

    n_samples = len(pseudo_labels)
    test_samples = n_samples

    prediction_vec = np.zeros(test_samples)

    err = 0
    predictions = []
    for i in range(0, test_samples):
            
        CheckLabelKnown(model, pseudo_labels[i])
        y_true_soft = LabelToActivation(pseudo_labels[i], model.label)
       
        h = model.W.shape[0]
        w = model.W.shape[1] 

        found_digit[np.argmax(y_true_soft)] += 1  # update the digit counter
            
        # PREDICTION
        #if(cntr>n_samples):
        if(True):
          y_pred_c = softmax(np.array(np.matmul(features[i,:], model.W) + model.b))
          prediction_vec[i] = np.argmax(y_pred_c)   

        y_pred_t = softmax(np.array(np.matmul(features[i,:], model.W_2) + model.b_2)) 
        

        # BACKPROPAGATION
        cost = y_pred_t-y_true_soft

        # Update weights
        for j in range(0,h):
            deltaW = np.multiply(cost, features[i,j])
            dW = np.multiply(deltaW, learn_rate)
            model.W_2[j,:] = model.W_2[j,:] - dW

        # Update biases
        db = np.multiply(cost, learn_rate)
        model.b_2 = model.b_2-db
        
        
        # If beginning of batch
        if(i % model_batch_size==0 and i!=0): 
            for k in range(0, w):
                if(found_digit[k]!=0):
                    tempW = np.multiply(model.W[:,k], found_digit[k])
                    tempB = np.multiply(model.b[k]  , found_digit[k])
                    model.W[:,k] = np.multiply(tempW+model.W_2[:,k], 1/(found_digit[k]+1))
                    model.b[k]   = np.multiply(tempB+model.b_2[k],   1/(found_digit[k]+1))
                    
            model.W_2  =  np.copy(model.W) # np.zeros((model.W.shape)) 
            model.b_2  =  np.copy(model.b) # np.zeros((model.b.shape))       
            found_digit = np.zeros(10)  # reset
        
        # the next part is only to plot the confusion matrix
        # if the train data is finished still train the model but do not save the results
        #if(cntr>n_samples):
        if(True):

            y_true_soft = LabelToActivation(pseudo_labels[i], model.label)
                   
            # Find the max iter for both true label and prediction
            # if(np.amax(y_true_soft) != 0):
            max_i_true = np.argmax(y_true_soft)

            #if(np.amax(y_pred_c) != 0):
            max_i_pred = np.argmax(y_pred_c)

            # Fill up the confusion matrix
            for k in range(0,len(model.label)):
                if(model.label[max_i_pred] == model.std_label[k]):
                    p = np.copy(k)
                if(model.label[max_i_true] == model.std_label[k]):
                    t = np.copy(k)

            model.conf_matr[t,p] += 1  
    
    return pseudo_labels, prediction_vec #, err_clu