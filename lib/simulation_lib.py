import numpy as np
import matplotlib.pyplot as plt 
import os

#######################################
#    ____  _     ___ _____ ____  
#   |  _ \| |   / _ \_   _/ ___| 
#   | |_) | |  | | | || | \___ \ 
#   |  __/| |__| |_| || |  ___) |
#   |_|   |_____\___/ |_| |____/ 
                              

""" Generates a bar plot of the class model given in input. The bar plot is generated from the attributo confusion_matrix """
def plot_barChart(model):
    
    conf_matr   = model.conf_matr
    title       = model.title 
    filename    = model.filename

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
def plot_confMatrix(model):

    title         = model.title 
    filename      = model.filename
    letter_labels = model.std_label 
    conf_matrix   = model.conf_matr    
    
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
def plot_table(model):

    title         = model.title 
    filename      = model.filename
    letter_labels = model.std_label 
    conf_matrix   = model.conf_matr   
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

######### Utility functions

def save_plots(model, SAVE_PATH, index):
    # Path(SAVE_PATH).mkdir(exist_ok=True) # Create directory if not exists
    os.makedirs(SAVE_PATH, exist_ok = True)

    # Create and save Figures
    table = plot_table(model)
    table.savefig(SAVE_PATH + '/Table_{}.png'.format(index))

    barchart = plot_barChart(model)
    barchart.savefig(SAVE_PATH + '/BarChart_{}.png'.format(index))

    cmtx = plot_confMatrix(model)
    cmtx.savefig(SAVE_PATH + '/CMtx_{}.png'.format(index))

    # Close all figures
    plt.close('all')




######### 
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

