# This program reads in both the onboard logs taken from the drone and the externally collected battery logs
# Then it runs through the external logs, creates the feature set, calculates accel/altitude, stores this to training data, and plots the results

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


# Read in onboard flight logs
# Onboard flight logs
# 19-07-07-02-09-59_FLY051.csv
# 19-07-06-08-02-41_FLY048
# 19-05-06-07-24-41_FLY042
# 19-09-28-03-28-52_FLY074 = 9/28-flight 1
# 19-09-28-04-07-04_FLY076 = 9/28-flight 2
data = pd.read_csv("19-09-28-04-07-04_FLY076.csv", usecols=(
1, 7, 8, 9, 10, 19, 20, 21, 22, 23, 34, 38, 45, 55, 56, 57, 58, 62, 63, 64, 65, 70, 71, 72, 73, 82, 83, 84, 85, 86, 87,
88, 89, 108, 110, 147, 148, 149, 150),
                   dtype=None, delimiter=',', low_memory=0)

data.columns = ['time', 'accelX', 'accelY', 'accelZ', 'accelComp',
                'velocityNorth', 'velocityEast', 'velocityDown', 'velocityComp', 'velocityHorizontal',
                'distanceTraveled', 'relativeHeight', 'actualHeight',
                'controllerAileron', 'controllerElevator', 'controllerRudder', 'controllerThrottle',
                'motorRPMFR', 'motorRPMFL', 'motorRPMBL', 'motorRPMBR', 'motorPPMFR', 'motorPPMFL', 'motorPPMBL',
                'motorPPMBR',
                'motorVoltsFR', 'motorVoltsFL', 'motorVoltsBL', 'motorVoltsBR',
                'motorCurrentFR', 'motorCurrentFL', 'motorCurrentBL', 'motorCurrentBR', 'accel2y', 'accel2comp',
                'motorPWMFR', 'motorPWMFL', 'motorPWMBL', 'motorPWMBR']

data.replace(['NaN', 'NaT'], np.nan, inplace=True)
data = data.dropna()

data['motorCurrentTotal'] = data['motorCurrentFR'] + data['motorCurrentFL'] + data['motorCurrentBL'] + data[
    'motorCurrentBR']
data['motorVoltsTotal'] = (data['motorVoltsFR'] + data['motorVoltsFL'] + data['motorVoltsBL'] + data[
    'motorVoltsBR']) / 4
data['RPMTotal'] = data['motorRPMFR'] + data['motorRPMFL'] + data['motorRPMBL'] + data['motorRPMBR']

pd.set_option('display.max_columns', 26)
pd.set_option('display.width', 400)
pd.set_option('display.max_rows', 100)
# data = data.quantile([0.02, .98])

# Remove extremities and irregularities of rpm
q1 = data['RPMTotal'].quantile(0.2)
q3 = data['RPMTotal'].quantile(0.8)
IQR = q3 - q1

filter = (data['RPMTotal'] >= q1 - 1.5 * IQR) & (data['RPMTotal'] <= q3 + 1.5 * IQR)
data = data.loc[filter]

# For externally collected cumulative current and voltage readings from drone battery using arduino logs
time = []
voltage = []
current = []

peaks = pd.DataFrame(columns=['time', 'current'])
craters = pd.DataFrame(columns=['time', 'current'])
# time offsets for:
# LOG = 364.5
# log2_9-28 = flight_076 = offset is 41.5 seconds
timeOffest = 41.5
with open('LOG2_9-28.txt') as infile:
    reader = csv.reader(infile)
    header = next(reader)
    j = 8
    for row in reader:
        if j == 8:
            j=0
            t = ((float(row[0]) / 1000) - timeOffest)
            if t > 0:
                time.append(t)
                voltage.append((int(row[1]) * 5 / 1024) / (7500 / (37500)))     # equation to convert analog reading into voltage
                current.append(((int(row[2])) - 510) * 5 / 1024 / 0.04 - 0.04)  # convert analog current into amps
        else:
            j += 1

# Butterworth signal filtering to smooth out raw arduino readings
b, a = signal.butter(4, 0.01, btype='lowpass')
c, d = signal.butter(8, 0.08, btype='lowpass')
filtCurrent = signal.filtfilt(b, a, data['motorCurrentTotal'], padlen=0)
filtVoltage = signal.filtfilt(b, a, data['motorVoltsTotal'], padlen=0)
filtAccel = signal.filtfilt(c, d, data['accelComp'], padlen=0)
filtAccel2 = signal.filtfilt(c, d, data['accel2comp'], padlen=0)
filtLogCurrent = signal.filtfilt(c, d, current, padlen=0)
filtLogVoltage = signal.filtfilt(c, d, voltage, padlen=0)

# State: 0 for unknown, 1 for takeoff, 2 for hover, 3 for climb, 4 for descend. Change this to enum later
state = 0
peaks = [[0, 0]]
craters = [[0, 0]]
peaksfull = [[0, 0]]
cratersfull = [[0, 0]]
maxPeak = 0
minCrater = 0
total = 0
avg = 0
buffer = pd.DataFrame(columns=['time', 'current'])
possibleMax = 0
rising = True
peakFound = False
climbStart = 0
descendstart = 0
accel = [[0, 0]]
veloc = [[0, 0]]
alt = [[0, 0]]
takeoffStart = 999999
prev_time = 0
RMSE = 0

outfile = open('Training_Data.csv', 'w')
outfile.write("Current, Voltage, MR_Crater, MR_Peak, MinCrater, MaxPeak, Average, Rising, State, Calculated_Acceleration, Logged_Acceleration\n")

for x in range(0, 400):
    buffer = buffer.append({'time': 0, 'current': 0}, ignore_index=True)


# parse through external arduino log with a moving time buffer window
# track current peaks/craters/averages, and detect state based on current behavior
# time range to track(takes longer to run the bigger this range is)
for i in range(300, 15000):
    # get new reading, drop old, find average
    if len(buffer) > len(buffer)-2:
        total = total - buffer.loc[0].current
        buffer = buffer.drop(buffer.index[0])
    buffer = buffer.append({'time': time[i], 'current': filtLogCurrent[i]}, ignore_index=True)
    total = total + filtLogCurrent[i]
    avg = total / len(buffer['time'])

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
                                #print("------------TAKEOFF: {}-{}".format(takeoffStart, takeoffEnd))
                                peaks = [[0, 0]]
                                craters = [[0, 0]]
                                peaksfull = [[0, 0]]
                                cratersfull = [[0, 0]]

                                maxPeak = 0
                                minCrater = 0
                                del buffer
                                buffer = pd.DataFrame(columns=['time', 'current'])
                                buffer = buffer.append({'time': time[i], 'current': filtLogCurrent[i]},
                                                       ignore_index=True)
                                total = filtLogCurrent[i]
                                avg = total / len(buffer)
                                alt.append([time[i], 2])
                                #print("-----------------or HOVER START: {}".format(takeoffEnd))

    # if the drone is climbing, mapp current to acceleration
    if state == 3:
        accelMapping = ((filtLogCurrent[i] - .505) / 10)  # the original value was -.5, the -.505 is from trial/error
        # scaling for increasing accelerations(improves accuracy)
        if (accelMapping > 1):
            accelMapping = (accelMapping * ((1 + (1 / accelMapping)) / 2))

        accelMapping = accelMapping - 1

        accel.append([time[i], (accelMapping)])
        #sampleTime = 0.002 # for 400hz
        #sampleTime = 0.016 # for 50hz
        timePeriod = time[i] - prev_time
        veloc.append([time[i], ((accel[len(accel) - 1][1]) * timePeriod + (veloc[len(veloc) - 1][1]))])
        # find new altitude derived from acceleration
        alt.append([time[i], (((veloc[len(veloc) - 1][1]) * timePeriod) + alt[len(alt) - 1][1] + (0.5 * (accel[len(accel) - 1][1]) * timePeriod**2))])  # velocity should be multiplied by the time window of 0.002, the 0.012 is just from trial and error(see physics equations
        # ascending/descending ends once current average is back in 'hover' levels with close peaks
        if 10.2 < avg < 11.2 and (maxPeak - minCrater < 2):
           #print("-----------------CLIMB ENDED: {}".format(craters[len(craters) - 1][0]))
            veloc.append([time[i], 0])
            alt.append([time[i], alt[len(alt) - 1][1]])

    if 10.2 < avg < 11.2 and state != 2:
        if (maxPeak - minCrater < 2):
           # print(
           #     "-----------------HOVER START: {}".format(max(craters[len(craters) - 1][0], peaks[len(peaks) - 1][0])))
            accel.append([time[i], 0])
            veloc.append([time[i], 0])
            alt.append([time[i], alt[len(alt) - 1][1]])
            state = 2
    if state == 2:
        accel.append([time[i], 0])
        veloc.append([time[i], 0])
        alt.append([time[i], alt[len(alt) - 1][1]])
        if (maxPeak - minCrater > 2 and minCrater != 0):
            # print("-----------------HOVER END: {}".format(time[i]))
            #print("-----------------HOVER END: {}".format(craters[len(craters) - 1][0]))
            climbStart = time[i]
            #print("-----------------CLIMB START: {}".format(climbStart))
            state = 3

    # remove peaks/craters outside of time window
    if ((time[i] - float(peaks[0][0])) > 4):
        if len(peaks) > 1:
            del peaks[0]
            maxPeak = max(peaks, key=lambda x: x[1])[1]
    if ((time[i] - craters[0][0]) > 4):
        if len(craters) > 1:
            del craters[0]
            minCrater = min(craters, key=lambda x: x[1])[1]

    # detect peaks/craters
    if rising:
        if filtLogCurrent[i - 1] < filtLogCurrent[i]:
            rising = True
        elif abs(craters[len(craters) - 1][1] - filtLogCurrent[i - 1]) > 0.3:
            peakFound = True
            peaks.append([time[i - 1], filtLogCurrent[i - 1]])
            peaksfull.append([time[i - 1], filtLogCurrent[i - 1]])

            maxPeak = max(peaks, key=lambda x: x[1])[1]
            rising = False
    else:
        if filtLogCurrent[i - 1] > filtLogCurrent[i]:
            rising = False
        elif abs(peaks[len(peaks) - 1][1] - filtLogCurrent[i - 1]) > 0.3:
            peakFound = False
            craters.append([time[i - 1], filtLogCurrent[i - 1]])
            cratersfull.append([time[i - 1], filtLogCurrent[i - 1]])
            minCrater = min(craters, key=lambda x: x[1])[1]
            rising = True

    if filtLogCurrent[i - 1] <= filtLogCurrent[i] and not peakFound:
        possibleMax = filtLogCurrent[i]
    elif filtLogCurrent[i - 1] >= filtLogCurrent[i] and peakFound:
        possibleMax = filtLogCurrent[i]

    prev_time = time[i]

    # Add the feature set for this time period to training data
    if time[i] > takeoffStart +1:
        line = str(filtLogCurrent[i]) + ", " + str(voltage[i]) + ", " + str(craters[-1][1]) + ", " + str(peaks[-1][1]) + ", " \
               + str(minCrater) + ", " + str(maxPeak) + ", " + str(avg) + ", " + str(int(rising)) + ", " + str(state) + ", " + str(accel[-1][1])
        index = data['time'].sub(time[i]).abs().idxmin()
        targetAccel = data['accelComp'].loc[index]
        targetAccel -= 1
        #accel.append([time[i], (accelMapping)])
        line += ", " + str(targetAccel) + "\n"
        outfile.write(line)

        if i > 10534 and i < 14033:
            print(i)
            RMSE += (targetAccel - accel[-1][1])**2

outfile.close()

RMSE = (RMSE)/3500
print("RMSE:", RMSE)

p = np.array(peaksfull)
c = np.array(cratersfull)

# data.plot(x='time', y='motorCurrentTotal')
#peaks.plot(kind='scatter', x='time', y='current', ax = ax)
accelT = [x[0] for x in accel]
accelC = [x[1] for x in accel]
veloT = [x[0] for x in veloc]
veloV = [x[1] for x in veloc]
altT = [x[0] for x in alt]
altY = [x[1] for x in alt]

# plt.plot(veloT, veloV, label="Calculated Velocity", alpha=1, linewidth=1.4)

# plt.plot(time, current, label="Raw Current", alpha=.8, linewidth=1)
plt.plot(time, current, label="Raw Current", alpha=1, linewidth=.8)
plt.plot(time, filtLogCurrent, label="Filtered Current", alpha=1, linewidth=1.3)

#plt.plot(accelT, accelC,  label="Calculated Acceleration", alpha=1, linewidth=1.4)#change colors back before uncommenting

plt.plot(data['time'],data['accelComp']-1, label="Logged Acceleration", alpha=1, linewidth=1.4, color='g') #change colors back to m/g

# plt.plot(data['time'],data['accel2comp']-1, label="Logged Acceleration2", alpha=1, linewidth=1.4, color='b')#change colors back to m/g

# plt.plot(data['time'],data['accel2y'], label="y2", alpha=1, linewidth=1.4, color='b')#change colors back to m/g
# plt.plot(data['time'],data['accelY'], label="y", alpha=1, linewidth=1.4, color='b')#change colors back to m/g

# plt.plot(altT, altY, label="Calculated Altitude", alpha=1, linewidth=1.4, color='m')
#plt.plot(data['time'],data['relativeHeight'], label="Logged Altitude", alpha=1, linewidth=1.4, color='g')

plt.plot(time,voltage, label="Voltage", alpha=.8, linewidth=.8, color='m')
# plt.plot(time, filtLogVoltage, label= "Filtered Voltage", alpha=.8, linewidth=1)

# plt.plot(data['time'],filtCurrent, label="Current", alpha=1, linewidth=2)
# plt.plot(data['time'],data['motorCurrentTotal'], label="Current", alpha=1, linewidth=2)

# plt.plot(data['time'],data['distanceTraveled']/10, label="distanceTraveled", alpha=1, linewidth=2)
# plt.plot(data['time'],data['velocityComp']*10, label="Logged Velocity", alpha=1, linewidth=1.4)

# plt.plot(data['time'],data['motorCurrentFR'], label="CurrentFR", alpha=.8, linewidth=1.2)
# plt.plot(data['time'],data['motorCurrentBR'], label="CurrentBR", alpha=.8, linewidth=1.2)
# plt.plot(data['time'],data['motorCurrentFL'], label="CurrentFL", alpha=.8, linewidth=1.2)
# plt.plot(data['time'],data['motorCurrentBL'], label="CurrentBL", alpha=.8, linewidth=1.2)

# plt.plot(data['time'],data['accelComp']-1, label="Logged Acceleration", alpha=1, linewidth=1.4)

plt.xlabel('Time(sec)', fontsize=12)
# plt.ylabel('Acceleration(m/s^2)', fontsize=12)
plt.ylabel('Current(A), Acceleration(10*m/s^2), Altitude(m)', fontsize=12)
#plt.ylabel('Current(A), Acceleration(10*m/s, Height(m)', fontsize=12)
#plt.ylabel('Acceleration(m/s^2)', fontsize=12)
# plt.ylabel('Current(A)', fontsize=12)

# plt.plot(data['time'],filtVoltage, label="Voltage", alpha=1, linewidth=2)
# plt.plot(data['time'],filtAccel*10, label="Accel", alpha=1, linewidth=2)
# plt.plot(data['time'],data['accelComp']*10, label="AccelRaw", alpha=1, linewidth=2)
plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2, borderaxespad=0, frameon=False)
# plt.legend(loc=' center right', bbox_to_anchor=(1, 0.5), ncol=1, borderaxespad=0, frameon=False)
plt.legend()

plt.show()
