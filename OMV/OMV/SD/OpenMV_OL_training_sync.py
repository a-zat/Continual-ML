import sensor, image, ustruct #, nn_st
import ulab
from ulab import numpy as np
from pyb import USB_VCP
import OpenMV_myLib as myLib
import gc
import tf  # <------- If we use the tflite model on the SD memory
import pyb
import os
import io
import uarray
import sys

#################################
# INITIALIZE CAMERA

usb = USB_VCP()
sensor.reset()                          # Reset and initialize the sensor.
sensor.set_contrast(3)
sensor.set_brightness(0)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_pixformat(sensor.GRAYSCALE)  # Set pixel format to Grayscale
sensor.set_framesize(sensor.QQQVGA)     # Set frame size to 80x60
sensor.skip_frames(time = 2000)         # Wait for settings take effect.

#################################

'''
bdev = zephyr.DiskAccess('SDHC')        # create block device object using DiskAccess
os.VfsFat.mkfs(bdev)                    # create FAT filesystem object using the disk storage block
os.mount(bdev, '/sd')                   # mount the filesystem at the SD card subdirectory
with open('/sd/hello.txt','w') as f:    # open a new file in the directory
    f.write('Hello world')
    f.close()              # write to the file
print(open('/sd/hello.txt').read())     # print contents of the file
'''

#net = nn_st.loadnnst('network')         # [CUBE.AI] Initialize the network
net = tf.load("OMV_Pruned_cnn.tflite", load_to_fb=True)
nn_input_sz = 28                        # The CNN input is 28x28

OL_layer = myLib.LastLayer()            # Create class for the training

myLib.load_biases(OL_layer)             # Read from the txt file the weights and save them
myLib.load_weights(OL_layer)            # Read from the txt file the biases and save them




# TRAINING METHOD SELECTION **********************************
# 0 -> no training, just inference
# 1 -> OL               WORKS - performs good
# 2 -> OLV2             WORKS - performs good
# 3 -> LWF              WORKS - performs good
# 4 -> CWR              WORKS - performs good
# 5 -> OL mini batch    WORKS - performs good
# 6 -> OLV2 mini batch  WORKS - performs good
# 7 -> LWF mini batch   WORKS - performs good - careful around label 30 camera reboots easily , dunno why
# 8 -> MY ALGORITHM
OL_layer.method = 8




myLib.allocateMemory(OL_layer)

label = 'X'

# DEFINE TRAINING PARAMS
OL_layer.l_rate      = 0.005 #  0.00001  per MY ALG
OL_layer.batch_size  = 256
OL_layer.train_limit = 4000     # after how many prediction start testing
OL_layer.counter     = 0        # just a reset
midpoint_type = 1

led1 = pyb.LED(2)
led2 = pyb.LED(3)
#out_feat = uarray.array()

counter = 0


while(True):


    label_b = usb.recv(1, timeout=5000)         # Receive the label from the laptop
    cmd_b   = usb.recv(4, timeout=10)           # Receive the command message from the laptop

    label = label_b.decode("utf-8")             # convert from byte to string
    cmd   = cmd_b.decode("utf-8")

    '''
    # extra info from the pc: when to close the feature file
    cmd_b_2 = usb.recv(4, timeout=10)
    cmd2    = cmd_b_2.decode("utf-8")
    cmd_b_3 = usb.recv(6, timeout=10)
    cmd3    = cmd_b_3.decode("utf-8")
    '''

    # STREAM
    if(cmd == 'snap'):

        img = sensor.snapshot()                 # Take the photo and return image

        if(OL_layer.counter>OL_layer.train_limit):
            myLib.write_results(OL_layer)       # Write confusion matrix in a txt file

        img = img.compress()
        usb.send(ustruct.pack("<L", img.size()))
        usb.send(img)

    # STREAM BUT SHOW HOW THE CAMERA MANIPULATES THE IMAGE BEFORE INFERENCE
    elif(cmd == 'elab'):

        img = sensor.snapshot()                 # Take the photo and return image
        led1.toggle()
        # Now I write some code for creating a txt file inside the SD memory
        # Seems working if I set a condition, the filename has a '/' and it is in append mode
        # Notice that to see the file the OpenMV has to be unplugged and plugged in
        #if img:
        #    with open('/Prova_elab.txt', 'a') as f:
        #        f.write('scrivo')
        #        f.write('\n')

        img.midpoint(midpoint_type, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,
        img = img.compress()
        usb.send(ustruct.pack("<L", img.size()))
        usb.send(img)


        #usb.send(b'daOMV')


    #out_frozen_old = []
    #out_frozen = []
    # TRAIN

    elif(cmd == 'trai'):

        t_0 = pyb.millis()

        img = sensor.snapshot()             # Take the photo and return image
        img.midpoint(midpoint_type, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,

        #if img:   #DEBUG: check if the image is captured
        #    led2.toggle()

        out_frozen = net.classify(img)       # Run the inference on frozen model
        outputf = out_frozen[0].output()
        #out_feat = out_feat.append(counter)

        #out_frozen = net.predict(img)

        #if out_frozen: #and (counter%50 == 0 or counter%51 == 0):  #DEBUG: check if the prediction is performed
        #    led2.toggle()
        strout = ''

        for i in range(0, len((out_frozen)[0].output())):
            if outputf[i] == 0.0:   #out_frozen[i] == 0.0:
                str1 = '0.000'
            else:
                str1 = '%.3f'%outputf[i]     #'%.3f'%out_frozen[i]
            strout = strout + str1 + ','

        #if len(strout) != 0:
        #    led2.toggle()
        usb.send(strout)

        #if out_frozen == out_frozen_old:
        #    led2.toggle()

        #out_frozen_old = out_frozen

        #reslt = 'prediction done'
        #output = io.StringIO()



        #if out_frozen:    #DEBUG: check if the prediction is performed and if the creation
            #led2.toggle() #       of the file is a problem

        #if counter == 0:
        #with open('Prova_trai.txt', 'w') as f:
        #    f.write( str(out_frozen[0].output()) )
        #    f.write("\n")
            #led2.toggle()
        #else:
        #if counter !=0:

        #with open('Prova_trai.txt', 'w') as f:
        #    f.write( str(out_frozen[0].output()) )
        #    f.write("\n")

        #np.savetxt('Prova_trai.txt', out_frozen, fmt='%.3f')


        #else:
        #    myled1.on()



        #if counter < 0:
        #    myled.ON()

        #if out_frozen and counter%600 == 0:
        #    with open('/Prova_trai_mid.txt', 'w') as f:
        #        f.write( str(out_frozen) )

        #if out_frozen and counter%1200 == 0:
        #    with open('/Prova_trai_end.txt', 'w') as f:
        #        f.write( str(out_frozen) )

        '''

        #print(out_frozen)

        t_1 = pyb.millis()

        # CHECK LABEL
        myLib.check_label(OL_layer, label)
        true_label = myLib.label_to_softmax(OL_layer, label)

        # PREDICTION - BACK PROPAGATION
        prediction = myLib.train_layer(OL_layer, true_label, out_frozen)

        t_2 = pyb.millis()

        # Update confusion matrix
        if(OL_layer.counter>OL_layer.train_limit):
            myLib.update_conf_matr(true_label, prediction, OL_layer)

        OL_layer.times[0,0] += t_1 - t_0
        OL_layer.times[0,1] += t_2 - t_1
        OL_layer.times[0,2] += t_2 - t_0
        OL_layer.counter += 1

        if(OL_layer.counter>OL_layer.train_limit-4):
            myLib.write_results(OL_layer)       # Write confusion matrix in a txt file

        if(cmd3 == 'allert'):
            usb.send(output.write(reslt), timeout = 150)

        if(cmd2 == 'endt'):
            file_features.close()

        myLib.prova(OL_layer)
        '''

    # STREAM
    else:
        img = sensor.snapshot()             # Take the photo and return image

    counter += 1


    #img = sensor.snapshot()                 # Take the photo and return image
    #img.midpoint(midpoint_type, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,

    #img = img.compress()
    #usb.send(ustruct.pack("<L", img.size()))
    #usb.send(img)





