import sensor
import ustruct
import nn_st
import pyb
from pyb import USB_VCP

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