path = 'Balanced_data'
from helper_functions import X_set,y_set
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import numpy as np

plt.rcParams['lines.linewidth'] = 2.5


def fft(sample_sensor):
    fs = 1/1000
    fourier = np.fft.fft(sample_sensor)
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    power_spectrum = np.log(power_spectrum)
    return power_spectrum, np.abs(freqs)

def fourier_signal_normalization_harmonics(sample):
    
    amp= fft(sample)[0]
    freq= fft(sample)[1]
    amp_list =[]
    freq_list =[]

    for i in range(170,250):
    #for i in range(0,len(amp)):
        amp_list.append(amp[i])
        freq_list.append(freq[i])

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq

def fft_log_plot(index):
    __,s2,s3,s4,__ = X_set(path,'none')
    s2 = fourier_signal_normalization_harmonics(s2[index])
    s3= fourier_signal_normalization_harmonics(s3[index])
    s4= fourier_signal_normalization_harmonics(s4[index])

    '''defect = y_set(path)['defect'][index]
    if defect =='df':defect ='Fiber failure'
    if defect =='dm':defect ='Matrix failure'
    if defect =='dd':defect ='Delamination'
    plt.title(f'Harmonics in log scale ({defect})')


    plt.plot(s2[1],s2[0])
    plt.plot(s3[1],s3[0])
    plt.plot(s4[1],s4[0])
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Amplitude (v)')'''
  
    fig,axs = plt.subplots(3,sharex=True,sharey=True)
    defect = y_set(path)['defect'][index]
    if defect =='df':defect ='Fiber failure'
    if defect =='dm':defect ='Matrix failure'
    if defect =='dd':defect ='Delamination'
    fig.suptitle(f'FFT of signal in log scale ({defect})')
    fig.text(0.5 , 0.04, 'Frequency (kHz)',ha = 'center')
    fig.text(0.04 , 0.5, 'Amplitude (v)',va = 'center',rotation = 'vertical')

    axs[0].plot(s2[1],s2[0])
    axs[0].grid()
    axs[0].set_title('s2')

    axs[1].plot(s3[1],s3[0])
    axs[1].grid()
    axs[1].set_title('s3')

    axs[2].plot(s4[1],s4[0])
    axs[2].grid()
    axs[2].set_title('s4')


    plt.show()



'''
dd --> 0
df --> 1
all --> 54
dm --> 58

'''
'''fft_log_plot(0)
fft_log_plot(1)
fft_log_plot(54)
fft_log_plot(58)'''


__,s2,s3,s4,__ = X_set(path,'none')
dd = fourier_signal_normalization_harmonics(s3[0])
df = fourier_signal_normalization_harmonics(s3[1])
all = fourier_signal_normalization_harmonics(s3[54])
dm = fourier_signal_normalization_harmonics(s3[58])
fig,axs = plt.subplots(4,sharex=True,sharey=True)


fig.suptitle(f'Harmonics in log scale of sensor 2')
fig.text(0.5 , 0.04, 'Frequency (kHz)',ha = 'center')
fig.text(0.04 , 0.5, 'Amplitude (v)',va = 'center',rotation = 'vertical')

axs[0].plot(df[1],df[0])
axs[0].grid()
axs[0].set_title('Fiber failure')

axs[1].plot(dm[1],dm[0])
axs[1].grid()
axs[1].set_title('Matrix failure')

axs[2].plot(dd[1],dd[0])
axs[2].grid()
axs[2].set_title('Delamination')

axs[3].plot(all[1],all[0])
axs[3].grid()
axs[3].set_title('All defect modes')


plt.show()