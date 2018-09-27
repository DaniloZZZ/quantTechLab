
# coding: utf-8

# In[1]:


import serial
from serial import SerialException


# In[2]:


try:
    serialData = serial.Serial('/dev/ttyUSB0',115200)

    try:
        while True:
            line = serialData.readline()
            value = -1
            if line[-2:] != b'\r\n':
                print("Something wrong with data format:",line,"expected 'b\r\n'")
            else:
                value = int(line[:-2])
                print(value)
    except SerialException as e:
        print("ERR",e)
except SerialException as e:
    print(e)
