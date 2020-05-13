# This program connects to the drone microcontroller via bluetooth and performs accel and altitude prediction based on battery readings

import numpy as np
import bluetooth as bluetooth

# Detect all Bluetooth devices and Create an array with all addresses
print("Searching for devices...")
nearby_devices = bluetooth.discover_devices(lookup_names=True)
print(nearby_devices)
# Run through all the devices found and list their name
num = 0
print("Select your device by entering its coresponding number...")
for i in range(len(nearby_devices)):
    num += 1
    print(str(num) + ": " + str(nearby_devices[i]))

# Allow the user to select their Arduino
selection = int(input("> ")) - 1
bd_addr = str(nearby_devices[selection][0])
port = 1
print((bd_addr, port))

sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((bd_addr, port))
print("Connected")
for i in range(200):
    data = sock.recv(524288)

from tkinter import *

tk = Tk()
a = Label(tk, text="Estimated Altitude(m):", anchor=NW, justify=LEFT, width=40, font=("Lucida Console", 60))
ac = Label(tk, text="Estimated Acceleration(m/s^2):", anchor=W, justify=LEFT, width=40, font=("Lucida Console", 60))
cu = Label(tk, text="Measured Current(A):", anchor=SW, justify=LEFT, width=40, font=("Lucida Console", 60))
vo = Label(tk, text="Measured Voltage(V):", anchor=SW, justify=LEFT, width=40, font=("Lucida Console", 60))
a.pack()
ac.pack()
cu.pack()
vo.pack()
tk.update_idletasks()
tk.update()

time = 0
current = 0
peaks = [[0, 0]]
craters = [[0, 0]]
state = 0
maxPeak = 0
minCrater = 0
total = 0
avg = 0
buffer = np.zeros((70, 2), dtype=int)
prev_current = 0
prev_time = 0
possibleMax = 0
rising = True
peakFound = False
climbStart = 0
descendstart = 0
accel = []
veloc = [[0, 0], [0,0]]
alt = [[0, 0], [0,0]]
takeoffStart = 999999
ascending = False
descending = False
sign = "+"
string = ""
vals = []
data = ""
string = ""
vals = []

# NOTE
# This is receiving at around 23ms
while True:
    data = sock.recv(1024)
    string += str(data.decode('utf-8'))
    data_end = string.find('\n')
    if data_end != -1:
        string = string[:data_end]
        vals = string.split(", ")
        if len(vals) > 2:
            if(vals[0] is not ''):
                time = ((float(vals[0])) / 1000)
            else:
                time = prev_time+0.03
            voltage =  (int(vals[1]) * 5 / 1024) / (7500 / (37500))  # equation to convert analog reading into voltage
            current = (((int(vals[2])) - 510) * 5 / 1024 / 0.04 - 0.04)

            if len(buffer) > 68:
                total = total - buffer[0][1]
                buffer = buffer[1:]
            buffer = np.vstack((buffer, np.array([time, current])))

            total = total + current
            avg = total / len(buffer)

            # takeoff sequence detection
            if 4 < avg < 8 and state == 0:
                if (1 < len(peaks) < 4) and (2 < len(craters) < 5):
                    if (4 < peaks[0][1] < 6) and (10 < peaks[1][1] < 14):
                        if (2.5 < peaks[1][0] - peaks[0][0] < 3.2):  # peaks pass
                            if (1.2 < craters[0][1] < 2) and (1.2 < craters[1][1] < 2.2):
                                if (1 < craters[1][0] - craters[0][0] < 1.8):
                                    if (0.1 < craters[2][0] - peaks[1][0] < .5):
                                        state = 2  # drone in takeoff.
                                        takeoffStart = craters[0][0]
                                        takeoffEnd = craters[2][0]
                                        print("------------TAKEOFF: {}-{}".format(takeoffStart, takeoffEnd))
                                        peaks = [[0, 0]]
                                        craters = [[0, 0]]
                                        maxPeak = 0
                                        minCrater = 0
                                        del buffer
                                        buffer = np.zeros((1, 2), dtype=int)
                                        buffer[0] = np.array([time, current])

                                        total = total + current
                                        avg = total / len(buffer)
                                        alt.append([time, 2])  # check this value, this is the estimated hover height after takeoff
                                        print("-----------------or HOVER START: {}".format(takeoffEnd))

            # Current mapping for ascending/descending
            if state == 3:
                accelMapping = ((current - .505) / 10)
                if (accelMapping >= 1):
                    accelMapping = (accelMapping * (((1 + (1 / accelMapping)) / 2)))
                else:
                    accelMapping = accelMapping / 1.1
                accelMapping = accelMapping - 1

                accel.append([time, (accelMapping)])
                if not descending and minCrater < 8:
                    descending = True
                    ascending = False
                    print("Descending at ", min(craters, key=lambda x: x[1]))

                timePeriod = time - prev_time
                veloc.append([time, ((accel[len(accel) - 1][1] * timePeriod) + veloc[len(veloc) - 1][1])])
                alti = (alt[len(alt) - 1][1] + ((veloc[len(veloc) - 1][1]) * timePeriod) + 0.5 * (
                            accel[len(accel) - 1][1] * (timePeriod) ** 2))
                if alti <= 1:
                    alt.append([time, 1])
                elif alti >=3:
                    alt.append([time, 3])
                else:
                    alt.append([time, alti])
                if 10.2 < avg < 11.2 and (maxPeak - minCrater < 2):
                    print("-----------------CLIMB ENDED: {}".format(craters[len(craters) - 1][0]))
                    ascending = False
                    descending = False
                    veloc.append([time, 0])
                    alt.append([time, alt[len(alt) - 1][1]])
                    
            if 10.2 < avg < 11.2 and state != 2:
                if (maxPeak - minCrater < 2):
                    print("-----------------HOVER START: {}".format(
                        max(craters[len(craters) - 1][0], peaks[len(peaks) - 1][0])))
                    if state == 0:
                        alt.append([time, 2])
                    else:
                        alt.append([time, alt[len(alt) - 1][1]])
                    veloc.append([time, 0])
                    accel.append([time[i], 0])
                    state = 2

            elif state == 2:
                veloc.append([time, 0])
                accel.append([time[i], 0])
                alt.append([time, alt[len(alt) - 1][1]])
                if (maxPeak - minCrater > 2 and minCrater != 0):
                    # print("-----------------HOVER END: {}".format(time[i]))
                    print("-----------------HOVER END: {}".format(craters[len(craters) - 1][0]))
                    climbStart = time
                    print("-----------------CLIMB START: {}".format(climbStart))
                    ascending = True
                    state = 3

            # remove peaks/craters outside of time window
            if ((time - float(peaks[0][0])) > 2):
                if len(peaks) > 1:
                    # peaks = peaks.remove(0)
                    del peaks[0]
                    maxPeak = max(peaks, key=lambda x: x[1])[1]
            if ((time - craters[0][0]) > 2):
                if len(craters) > 1:
                    del craters[0]
                    minCrater = min(craters, key=lambda x: x[1])[1]

            # detect peaks/craters
            if rising:
                if prev_current < current:
                    rising = True
                elif abs(craters[len(craters) - 1][1] - prev_current) > 0.4:
                    peakFound = True
                    peaks.append([prev_time, prev_current])
                    maxPeak = max(peaks, key=lambda x: x[1])[1]
                    rising = False
            else:
                if prev_current > current:
                    rising = False
                elif abs(peaks[len(peaks) - 1][1] - prev_current) > 0.4:
                    peakFound = False
                    craters.append([prev_time, prev_current])
                    minCrater = min(craters, key=lambda x: x[1])[1]
                    rising = True


            if prev_current <= current and not peakFound:
                possibleMax = current
            elif prev_current >= current and peakFound:
                possibleMax = current

            prev_time = time
            prev_current = current
            tk.update_idletasks()
            tk.update()
            #a = Label(tk, text=str(alt[len(alt)-1][1]))
            a.configure(text="Estimated Altitude(m):         {:.2f}".format(alt[len(alt)-1][1]))
            if len(accel) > 1:
                if accel[len(accel) - 1][1] < 0:
                    ac.config(text="Estimated Acceleration(m/s^2): -{:.2f}".format(abs(accel[len(accel) - 1][1])))
                else:
                    ac.config(text="Estimated Acceleration(m/s^2):  {:.2f}".format(accel[len(accel) - 1][1]))
            cu.configure(text="Measured Current(A):           {:.2f}".format(current))
            vo.configure(text="Measured Voltage(V):           {:.2f}".format(voltage))

        string = ""




