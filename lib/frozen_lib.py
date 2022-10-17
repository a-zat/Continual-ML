# Library for Frozen model

import numpy as np
import matplotlib.pyplot as plt

import tempfile
import os
import zipfile

def hostiry_training_plot(model):
    hist_loss     = model.history['loss']
    hist_val_loss = model.history['val_loss']
    hist_acc      = train_hist.history['accuracy']
    hist_val_acc  = train_hist.history['val_accuracy']
    epoch_list    = list(range(epochs))
   
    plt.subplot(211)
    plt.plot(epoch_list, hist_acc,  label='Accuracy', linewidth=3)
    plt.plot(epoch_list, hist_val_acc,  label='Validation accuracy', linewidth=3)
    plt.legend(prop={'size': 17})
    plt.xlabel('Epochs',  fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)

    plt.subplot(212)
    plt.plot(epoch_list, hist_loss, 'bo', label='Training loss', linewidth=3)
    plt.plot(epoch_list, hist_val_loss, 'r', label='Validation loss', linewidth=3)
    plt.legend(prop={'size': 17})
    plt.xlabel('Epochs',  fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)

    ROOT_PATH = os.path.abspath('')
    PLOT_PATH = ROOT_PATH + "/Results/"
    plt.savefig(PLOT_PATH + 'training_History.jpg')
    plt.show()

def plot_ConfusionMatrix(conf_matrix):

    figure = plt.figure()
    axes = figure.add_subplot()

    label = ['0','1','2','3','4','5']

    caxes = axes.matshow(conf_matrix, cmap=plt.cm.Blues)
    figure.colorbar(caxes)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            axes.text(x=j, y=i,s=int(conf_matrix[i, j]), va='center', ha='center', size='large')

    axes.xaxis.set_ticks_position("bottom")
    axes.set_xticklabels([''] + label)
    axes.set_yticklabels([''] + label)

    plt.xlabel('PREDICTED LABEL', fontsize=15)
    plt.ylabel('TRUE LABEL', fontsize=15)
    ROOT_PATH = os.path.abspath('')
    PLOT_PATH = ROOT_PATH + "/Results/"
    plt.savefig(PLOT_PATH + 'training_ConfMatrix.jpg')
    plt.show()


def testing(data, label_lett, model):

    conf_matrix = np.zeros((6,6))

    prediction = model.predict(data)
    
    for i in range(0, data.shape[0]):

        # Find the max iter for both true label and prediction
        max_i_true = int(label_lett[i])

        max_i_pred = np.argmax(prediction[i,:])

        conf_matrix[max_i_true, max_i_pred] = conf_matrix[max_i_true, max_i_pred] + 1

    return conf_matrix


def plot_Accuracy(conf_matrix):
                        
    tot_cntr = 0
    correct_cntr = 0

    corr_ary   = np.zeros(6)
    tot_ary    = np.zeros(6)
    bar_values = np.zeros(7) 

    letter_labels = ['0','1','2','3','4','5','Model']
    blue2 = 'cornflowerblue'
    colors = [blue2, blue2, blue2, blue2, blue2, blue2, 'steelblue']

    for i in range(0, conf_matrix.shape[0]):
        bar_values[i] = round(round(conf_matrix[i,i]/ sum(conf_matrix[i,:]), 4)*100,2)
        tot_cntr += sum(conf_matrix[i,:])
        correct_cntr += conf_matrix[i,i]

    bar_values[-1] = round(round(correct_cntr/tot_cntr, 4)*100,2)
    

    fig = plt.subplots(figsize =(10, 6))

    bar_plot = plt.bar(letter_labels, bar_values, color=colors, edgecolor='grey')

    for p in bar_plot:
        height = p.get_height()
        xy_pos = (p.get_x() + p.get_width() / 2, 105)

        plt.annotate(str(height) + '%', xy=xy_pos, xytext=(0, 0), textcoords="offset points", ha='center', va='bottom', fontsize=15,  fontweight ='bold')

    plt.axhline(y = 100, color = 'gray', linestyle = (0, (5, 10)) ) # Grey line at 100 %

    # Text and labels
    ROOT_PATH = os.path.abspath('')
    PLOT_PATH = ROOT_PATH + "/Results/"
    plt.ylim([0, 119])
    plt.ylabel('Accuracy %', fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.xticks([r for r in range(len(letter_labels))], letter_labels, fontweight ='bold', fontsize = 15) # Write on x axis the letter name
    plt.savefig(PLOT_PATH + 'training_Test.jpg')
    plt.show()

    print(f"Total correct guesses  {sum(corr_ary)}  -> {round(round(sum(corr_ary)/sum(tot_ary), 4)*100,2)}%")


def plot_Table(conf_matrix):

    table_values = np.zeros([3,conf_matrix.shape[0]])

    for i in range(0, table_values.shape[1]):

        if(sum(conf_matrix[i,:])==0):   # if for avoiding division by 0 that generates NAN                                
            table_values[0,i] = 0
        else:
            table_values[0,i] = round(conf_matrix[i,i]/sum(conf_matrix[i,:]),2)      # ACCURACY

        if(sum(conf_matrix[:,i])==0):   # if for avoiding division by 0 that generates NAN
            table_values[1,i] = 0
        else:
            table_values[1,i] = round(conf_matrix[i,i]/sum(conf_matrix[:,i]),2)      # PRECISION 

        if((table_values[1,i]+table_values[0,i])==0):     # if for avoiding division by 0 that generates NAN
            table_values[2,i] = 0
        else:
            table_values[2,i] = round((2*table_values[1,i]*table_values[0,i])/(table_values[1,i]+table_values[0,i]),2)    # F1 SCORE

    
    fig, ax = plt.subplots(figsize=(6,4)) 
    ax.set_axis_off() 
    
    table = ax.table( 
        cellText = table_values,  
        rowLabels = ['Accuracy', 'Precision', 'F1 score'],  
        colLabels = ['0','1','2','3','4','5'], 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        rowLoc='right',
        loc ='center')   

    table.set_fontsize(14)
    ROOT_PATH = os.path.abspath('')
    PLOT_PATH = ROOT_PATH + "/Results/"
    plt.savefig(PLOT_PATH + 'training_Table.jpg')
    plt.show()


def plot_image(i, predictions_array, true_label, img):
    class_names = ['0','1','2','3','4','5']

    true_label, img = int(true_label[i]), img[i,:,:]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    img = (np.squeeze(img))## you have to delete the channel information (if grayscale) to plot the image
    plt.imshow(img, cmap="gray")

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)   


def plot_value_array(i, predictions_array, true_label):
    true_label = int(true_label[i])
    plt.grid(False)
    plt.xticks(range(6))
    plt.yticks([])
    thisplot = plt.bar(range(6), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')



#creation class with name person:
class save_info:
    #define init function to initialize main variables:
    def __init__(self, batch_size, epochs, metrics, optimizer,loss):

    #define main variables you want to store and use
        self.batch_size    = batch_size
        self.epochs     = epochs
        self.metrics     = metrics
        self.optimizer  = optimizer
        self.loss  = loss

def save_summary_model(model, MODEL_PATH, info, flag):
      with open(MODEL_PATH + 'model_summary.txt', "w") as new_file:
        new_file.write("PARAMETERS SAVED FROM THE TRAINING")
        match flag:
          case "original":
            new_file.write("\n\n This model has been trained for learning the first 6 digits from the MNIST dataset, this is the ORIGINAL MODEL")
          case "frozen":
            new_file.write("\n\n This model has been trained for learning the first 6 digits from the MNIST dataset, this is the FROZEN MODEL")
            new_file.write("\n")
            new_file.write("\n Batch size:       " + str(info.batch_size))
            new_file.write("\n Epochs:           " + str(info.epochs))
            new_file.write("\n Metrics:          " + str(info.metrics))
            new_file.write("\n Optimizer:        " + info.optimizer)
            new_file.write("\n Loss:             " + info.loss)
            new_file.write("\n\n")
          case _:
            raise ValueError("Unknown flag")
        model.summary(print_fn=lambda x: new_file.write(x + '\n'))
      new_file.close()


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)