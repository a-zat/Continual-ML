import sensor, image, ustruct, nn_st
import ulab
from ulab import numpy as np
from pyb import USB_VCP
import OpenMV_myLib as myLib
import gc
import pyb
import os
import io
import uarray
import sys
import lcd
usb = USB_VCP()
sensor.reset()
sensor.set_contrast(3)
sensor.set_brightness(0)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QQQVGA)
#sensor.set_framesize(sensor.QQVGA2)
sensor.skip_frames(time = 2000)
#lcd.init()
net = nn_st.loadnnst('network')
nn_input_sz = 28
led1 = pyb.LED(2)
led2 = pyb.LED(3)
counter = 0
while(True):
	cmd_b   = usb.recv(4, timeout=10)
	cmd   = cmd_b.decode("utf-8")
	if(cmd == 'snap'):
		img = sensor.snapshot()
		img = img.compress()
		usb.send(ustruct.pack("<L", img.size()))
		usb.send(img)
	elif(cmd == 'elab'):
		img = sensor.snapshot()
		#lcd.display(img)
		img = img.compress()
		usb.send(ustruct.pack("<L", img.size()))
		usb.send(img)
	elif(cmd == 'trai'):
		t_0 = pyb.millis()
		img = sensor.snapshot()
		#lcd.display(img)
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
	else:
		img = sensor.snapshot()
	counter += 1