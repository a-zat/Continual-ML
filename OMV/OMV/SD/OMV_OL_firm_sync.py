import sensor, image, ustruct, nn_st
import ulab
from ulab import numpy as np
from pyb import USB_VCP
import OpenMV_myLib as myLib
import gc
#import tf  # <------- If we use the tflite model on the SD memory
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

net = nn_st.loadnnst('network')         # [CUBE.AI] Initialize the network
nn_input_sz = 28                        # The CNN input is 28x28


led1 = pyb.LED(2)
led2 = pyb.LED(3)
#out_feat = uarray.array()

counter = 0


while(True):

    cmd_b   = usb.recv(4, timeout=10)           # Receive the command message from the laptop

    cmd   = cmd_b.decode("utf-8")

    # STREAM
    if(cmd == 'snap'):

        img = sensor.snapshot()                 # Take the photo and return image

        #if(OL_layer.counter>OL_layer.train_limit):
        #    myLib.write_results(OL_layer)       # Write confusion matrix in a txt file

        img = img.compress()
        usb.send(ustruct.pack("<L", img.size()))
        usb.send(img)

    # STREAM BUT SHOW HOW THE CAMERA MANIPULATES THE IMAGE BEFORE INFERENCE
    elif(cmd == 'elab'):

        img = sensor.snapshot()                 # Take the photo and return image
        #led1.toggle()

        #img.midpoint(midpoint_type, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,
        #img = img.compress()
        #usb.send(ustruct.pack("<L", img.size()))
        #usb.send(img)

        lcd.display(img)

    # TRAIN

    elif(cmd == 'trai'):

        t_0 = pyb.millis()

        img = sensor.snapshot()             # Take the photo and return image
        #img.midpoint(midpoint_type, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,

        out_frozen = net.predict(img)


        strout = ''

        for i in range (0, len(out_frozen)):
            if out_frozen[i] == 0.0:
                str1 = '0.000'
            elif out_frozen[i] >= 10 and out_frozen[i] < 100:
                str1 = '%.2f'%out_frozen[i]
            elif out_frozen[i] >= 100:
                str1 = '%.1f'%out_frozen[i]
            else:
                str1 = '%.3f'%out_frozen[i]
            strout = strout + str1 + ','


        usb.send(strout)

    # STREAM
    else:
        img = sensor.snapshot()             # Take the photo and return image
    '''
    img = sensor.snapshot()                 # Take the photo and return image
    #led1.toggle()

    #img.midpoint(midpoint_type, bias=0.5, threshold=True, offset=5, invert=True) # Binarize the image, size is 3x3,
    img = img.compress()
    usb.send(ustruct.pack("<L", img.size()))
    usb.send(img)
    '''
    counter += 1





