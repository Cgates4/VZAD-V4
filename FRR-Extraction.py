import numpy as np
import pandas as pd
from scipy.fft import rfft, fft, fftfreq
import matplotlib.pyplot as plt

allflashes = []
with open ("testing.txt", "r") as f:
    for l in range (5):
        next (f)
    for line in f:
        data = line.split()
        allflashes.append(float(data[1]))

dim = len (allflashes)
x = round (dim / 40)
allFos = []
allFms = []
allFvFms = []

for i in range (0, dim, 40):
    window = allflashes[i: i + 40]

    if len (window) >= 40:
        allFos.append(window[40-33]) #find Fo
        allFms.append(max(window)) #find Fm

for n in range (len(allFos)):
    allFvFms.append((allFms[n] - allFos[n]) / allFms[n])

numTrains = round (x/50)
organizedFvFm = []
for c in range (numTrains):
    train = []
    for r in range (50):
        train.append(allFvFms[(50*c)+r])
    organizedFvFm.append(train)

organizedFvFm_df = pd.DataFrame(organizedFvFm)
organizedFvFm_df = organizedFvFm_df.transpose()
organizedFvFm_df.to_csv('organizedflashtrains.csv', header = False, index = False)

organizedFm = []
for c in range (numTrains):
    Fm_train = []
    for r in range (50):
        Fm_train.append(allFms[(50*c)+r])
    organizedFm.append(Fm_train)

organizedFm_df = pd.DataFrame(organizedFm)
organizedFm_df = organizedFm_df.transpose()
organizedFm_df.to_csv('organizedFms.csv', header = False, index = False)

#get every 50th Fo and Fm
skip_Fos = allFos[::50]
skip_Fms = allFms[::50]

#remove first value from list because [::50] includes first value from skip list above
skip_Fos.pop(0)
skip_Fms.pop(0)

#get average value
avgFos = np.mean(skip_Fos)
avgFms = np.mean(skip_Fms)
#print (avgFos)
#print (avgFms)

#get average of 1st flash, 2nd flash, 3rd flash, etc
train_avg = organizedFvFm_df.mean(axis = 1)
train_avg = train_avg.values.tolist()

fft_points = len(train_avg)
x_axis = np.linspace(0, fft_points, endpoint=False)  #x axis indicators
x_fft = fftfreq(fft_points)[:fft_points // 2]  #make x axis to be half of number of total x values
y_fft = rfft(train_avg)  #perform fft on data
new_y_fft = y_fft / allflashes[7]
with open('FRR_FFT.txt', 'w') as f:
    for n in range(1,len(x_fft)):
        f.write(f"{x_fft[n]} \t {y_fft[n]}\n")

N = len (train_avg)
plt.plot(x_fft, (2 / N) * np.abs (y_fft[0 : N // 2]))
plt.grid()
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("FFR FFT Graph")
plt.savefig("FRR_FFT_Graph")


'''max_fft = 0
for Fo_loop in range (1, 40):
    allFos = []
    allFms = []
    allFvFms = []
    for i in range(0, dim, 40):
        window = allflashes[i: i + 40]

        if len(window) >= 40:
            allFos.append(window[Fo_loop])  # find Fo
            allFms.append(max(window))  # find Fm

    for n in range(len(allFos)):
        allFvFms.append((allFms[n] - allFos[n]) / allFms[n])

    numTrains = round (x / 50)
    organizedFvFm = []
    for c in range(numTrains):
        train = []
        for r in range(50):
            train.append(allFvFms[(50 * c) + r])
        organizedFvFm.append(train)

    organizedFvFm_df = pd.DataFrame(organizedFvFm)
    organizedFvFm_df = organizedFvFm_df.transpose()
    organizedFvFm_df.to_csv('organizedflashtrains.csv', header=False, index=False)

    organizedFm = []
    for c in range(numTrains):
        Fm_train = []
        for r in range(50):
            Fm_train.append(allFms[(50 * c) + r])
        organizedFm.append(Fm_train)

    organizedFm_df = pd.DataFrame(organizedFm)
    organizedFm_df = organizedFm_df.transpose()
    organizedFm_df.to_csv('organizedFms.csv', header=False, index=False)

    # get every 50th Fo and Fm
    skip_Fos = allFos[::50]
    skip_Fms = allFms[::50]

    # remove first value from list because [::50] includes first value from skip list above
    skip_Fos.pop(0)
    skip_Fms.pop(0)

    # get average value
    avgFos = np.mean(skip_Fos)
    avgFms = np.mean(skip_Fms)
    # print (avgFos)
    # print (avgFms)

    # get average of 1st flash, 2nd flash, 3rd flash, etc
    train_avg = organizedFvFm_df.mean(axis=1)
    train_avg = train_avg.values.tolist()

    fft_points = len(train_avg)
    x_axis = np.linspace(0, fft_points, endpoint=False)  # x axis indicators
    x_fft = fftfreq(fft_points)[:fft_points // 2]  # make x axis to be half of number of total x values
    y_fft = rfft(train_avg)  # perform fft on data

    new_y = y_fft[6:19]
    new_y_max = np.max(new_y)
    if new_y_max > max_fft:
        max_fft = new_y_max
        best_Fo = Fo_loop

        with open('Best_FRR_FTT.txt', 'w') as f:
            for n in range(1, len(x_fft)):
                f.write(f"{x_fft[n]} \t {y_fft[n]}\n")

        plt.clf() #clear graph

        N = len(train_avg)
        plt.plot(x_fft, (2 / N) * np.abs(y_fft[0: N // 2]))
        plt.grid()
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.title("FFT Graph")
        plt.savefig("Best_FRR_FFT_Graph")


print (best_Fo)
print (max_fft)'''
