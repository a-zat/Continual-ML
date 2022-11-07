import numpy as np
import matplotlib.pyplot as plt 
import os


##################################################################################################
#    _____                                    _                      _      ____  _       _       
#   |  ___| __ ___ _______ _ __    _ __   ___| |___      _____  _ __| | __ |  _ \| | ___ | |_ ___ 
#   | |_ | '__/ _ \_  / _ \ '_ \  | '_ \ / _ \ __\ \ /\ / / _ \| '__| |/ / | |_) | |/ _ \| __/ __|
#   |  _|| | | (_) / /  __/ | | | | | | |  __/ |_ \ V  V / (_) | |  |   <  |  __/| | (_) | |_\__ \
#   |_|  |_|  \___/___\___|_| |_| |_| |_|\___|\__| \_/\_/ \___/|_|  |_|\_\ |_|   |_|\___/ \__|___/
#                                                                                                 

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




######################################################################################
#       _        _   _             _                            ____  _       _       
#      / \   ___| |_(_)_   _____  | |    __ _ _   _  ___ _ __  |  _ \| | ___ | |_ ___ 
#     / _ \ / __| __| \ \ / / _ \ | |   / _` | | | |/ _ \ '__| | |_) | |/ _ \| __/ __|
#    / ___ \ (__| |_| |\ V /  __/ | |__| (_| | |_| |  __/ |    |  __/| | (_) | |_\__ \
#   /_/   \_\___|\__|_| \_/ \___| |_____\__,_|\__, |\___|_|    |_|   |_|\___/ \__|___/
#                                             |___/                                   
                              

""" Generates a bar plot of the class model given in input. The bar plot is generated from the attributo confusion_matrix """
def plot_barChart(model, id):

    match id:
        case 'model':
            conf_matr   = model.conf_matr
            title       = model.title + ' (Model)'
            #filename    = model.filename + '_model'
        case 'clust':
            conf_matr   = model.conf_matr2
            title       = model.title + ' (Clustering)'
            #filename    = model.filename + '_clust'

    real_label = ['0','1','2','3','4','5','6','7','8','9','Model']
    # Generate matrix of colors for the bars
    blue2 = 'cornflowerblue'
    colors = [blue2, blue2, blue2, blue2, blue2, 
              blue2, blue2, blue2, blue2, blue2, 'steelblue']  

    bar_values   = np.zeros(conf_matr.shape[0]+1)
    tot_pred     = 0
    correct_pred = 0

    # Compute the accuracy for each label and store it inside array
    for i in range(0, conf_matr.shape[0]):
        if( sum(conf_matr[i,:]) != 0):
            bar_values[i] = round(round(conf_matr[i,i]/sum(conf_matr[i,:]),4)*100, 2)      # Accuracy for each letter
        tot_pred += sum(conf_matr[i,:])
        correct_pred += conf_matr[i,i]

    bar_values[-1] = round(round(correct_pred/tot_pred, 4)*100,2)   # Overall accuracy of the model
    
    fig, ax = plt.subplots(figsize =(12, 8))
    bar_plot = plt.bar(real_label, bar_values, color=colors, edgecolor='grey')

    # Add text to each bar showing the percentage of accuracy
    for p in bar_plot:
        height = p.get_height()
        xy_pos = (p.get_x() + p.get_width() / 2, height)
        xy_txt = (0, -20) 

        # Avoid the text to be outside the image if bar is too low
        if(height>10):
            plt.annotate(str(height), xy=xy_pos, xytext=xy_txt, textcoords="offset points", ha='center', va='bottom', fontsize=12)
        else:
            plt.annotate(str(height), xy=xy_pos, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)

    
    # Plot
    plt.ylim([0, 100])
    plt.ylabel('Accuracy %', fontsize = 15)
    plt.xlabel('Classes', fontsize = 15)
    plt.xticks([r for r in range(len(real_label))], real_label, fontweight ='bold', fontsize = 12) # Write on x axis the letter name
    plt.title('Accuracy test - Method used: '+title, fontweight ='bold', fontsize = 15)

    return fig




""" Function that generates a plot showing the confusion matrix of the class given in input """
def plot_confMatrix(model, id):

    match id:
        case 'model':
            conf_matrix   = model.conf_matr
            title         = model.title + ' (Model)'
            #filename    = model.filename + '_model'
        case 'clust':
            conf_matrix   = model.conf_matr2
            title         = model.title + ' (Clustering)'
            #filename    = model.filename + '_clust

    letter_labels = model.std_label    
    
    fig = plt.figure(figsize =(6,6))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    res = ax.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
    width, height = conf_matrix.shape

    # Loop over data dimensions and create text annotations.
    for x in range(width):
        for y in range(height):
            ax.annotate(str(int(conf_matrix[x,y])), xy=(y, x), ha="center", va="center", size='large')

    cb = fig.colorbar(res)
    plt.xticks(range(width), letter_labels[:width])
    plt.yticks(range(height), letter_labels[:height])

    # labels, title and ticks
    plt.xlabel('PREDICTED LABELS')
    plt.ylabel('TRUE LABELS') 
    plt.title('OpenMV training confusion matrix - ' + title, fontweight ='bold', fontsize = 15)

    # plt.close(fig)
    return fig



""" Function that computes the accuracy, precision adn F1 score and generates a table """
def plot_table(model, id):

    match id:
        case 'model':
            conf_matrix   = model.conf_matr
            title         = model.title + ' (Model)'
            #filename    = model.filename + '_model'
        case 'clust':
            conf_matrix   = model.conf_matr2
            title         = model.title + ' (Clustering)'
            #filename    = model.filename + '_clust


    letter_labels = model.std_label 
    table_values  = np.zeros([3,conf_matrix.shape[1]])

    for i in range(0, table_values.shape[1]):
        if(sum(conf_matrix[i,:]) != 0):
            table_values[0,i] = round(conf_matrix[i,i]/sum(conf_matrix[i,:]),2)       # RECALL/SENSITIVITY

        if(sum(conf_matrix[:,i]) != 0):
            table_values[1,i] = round(conf_matrix[i,i]/sum(conf_matrix[:,i]),2)       # PRECISION 

        if((table_values[1,i]+table_values[2,i])!=0):
            table_values[2,i] = round((2*table_values[0,i]*table_values[1,i])/(table_values[0,i]+table_values[1,i]),2)  # F1 SCORE

    fig, ax = plt.subplots(figsize =(10, 3)) 
    ax.set_axis_off() 

    table = ax.table( 
        cellText = table_values,  
        rowLabels = ['Accuracy', 'Precision', 'F1 score'],  
        colLabels = letter_labels, 
        rowColours =["cornflowerblue"] * 200,  
        colColours =["cornflowerblue"] * 200, 
        cellLoc ='center',  
        loc ='upper left')         

    table.scale(1,2) 
    table.set_fontsize(10)
    ax.set_title('OpenMV training table - ' + title, fontweight ="bold") 

    # plt.close(fig) 
    return fig


############################################################################
#    _   _ _   _ _ _ _            __                  _   _                 
#   | | | | |_(_) (_) |_ _   _   / _|_   _ _ __   ___| |_(_) ___  _ __  ___ 
#   | | | | __| | | | __| | | | | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#   | |_| | |_| | | | |_| |_| | |  _| |_| | | | | (__| |_| | (_) | | | \__ \
#    \___/ \__|_|_|_|\__|\__, | |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#                        |___/                                              

def save_plots(model, SAVE_PATH, id, index):

    match id:
        case 'model':
            strid = '/model_'
        case 'clust':
            strid = '/clust_'

    # Path(SAVE_PATH).mkdir(exist_ok=True) # Create directory if not exists
    os.makedirs(SAVE_PATH + strid, exist_ok = True)

    # Create and save Figures
    table = plot_table(model, id)
    table.savefig(SAVE_PATH + strid + '/Table_{}.png'.format(index))

    barchart = plot_barChart(model, id)
    barchart.savefig(SAVE_PATH + strid +  '/BarChart_{}.png'.format(index))

    cmtx = plot_confMatrix(model, id)
    cmtx.savefig(SAVE_PATH +  strid + '/CMtx_{}.png'.format(index))

    # Close all figures
    plt.close('all')


import io
from PIL import Image


def save_plots2(model, SAVE_PATH, index):
    # Path(SAVE_PATH).mkdir(exist_ok=True) # Create directory if not exists
    # os.makedirs(SAVE_PATH, exist_ok = True)

    # Create and save Figures
    
    # Confusion matrix
    cmtx = plot_confMatrix(model)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_cmtx = Image.open(buf)
    plt.close(cmtx)

    # Accuracy chart
    barchart = plot_barChart(model)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_barchart = Image.open(buf)
    plt.close(barchart)

    # Metrics table
    table = plot_table(model)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_table = Image.open(buf)
    plt.close(cmtx)

    fig = plt.figure()
    fig.add_subplot(3, 1, 1, frameon=False)
    plt.imshow(img_cmtx)
    fig.add_subplot(3, 1, 2, frameon=False)
    plt.imshow(img_barchart)
    fig.add_subplot(3, 1, 3, frameon=False)
    plt.imshow(img_table)

