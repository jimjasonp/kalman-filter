import matplotlib.pyplot as plt
import numpy as np
path = 'random_data'

from helper_functions import X_set

plt.rcParams.update({'font.size': 16})

plt.rcParams['lines.linewidth'] = 2.5

__,s2,s3,s4,__ = X_set(path,'none')

s2 = s2[10]
s3 = s3[10]
s4 = s4[10]

'''fig,axs = plt.subplots(3,sharex=True,sharey=True)
fig.suptitle('Signal from all sensors (all defect modes)')
fig.text(0.5 , 0.04, 'time (ms)',ha = 'center')
fig.text(0.04 , 0.5, 'Electric potential (v)',va = 'center',rotation = 'vertical')

axs[0].plot(s2)
axs[0].grid()
axs[0].set_title('s2')

axs[1].plot(s3)
axs[1].grid()
axs[1].set_title('s3')

axs[2].plot(s4)
axs[2].grid()
axs[2].set_title('s4')

plt.show()'''

def fft(sample_sensor):
    fs = 1/1000
    fourier = np.fft.fft(sample_sensor)
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    power_spectrum = np.log(power_spectrum)
    return power_spectrum, np.abs(freqs)

s2 = fft(s2)
s3 = fft(s3)
s4 = fft(s4)

'''plt.plot(s2[1],s2[0],linestyle ='dashed', dashes = (10,15))
plt.plot(s3[1],s3[0],linestyle ='-.')
plt.plot(s4[1],s4[0],linestyle ='solid')
plt.grid()
plt.title('Signal with FFT transformation in log scale (all defect modes)')
plt.ylabel('Amplitude (v)')
plt.xlabel('Frequency (kHz)')
plt.legend(['s2','s3','s4'])
plt.show()'''



plt.title('Signal with FFT transformation in log scale of sensor 3 (all defect modes)')
plt.xlabel( 'Frequency (kHz)')
plt.ylabel('Amplitude (v)')
plt.plot(s3[1],s3[0])
plt.grid()
plt.annotate('Excitation frequency (125kHz)', xy=(125,-4), xytext=(200, -3),
arrowprops=dict(facecolor='black', shrink=0.2))
plt.annotate('Dominant harmonic (250 kHz)', xy=(250,-10), xytext=(120,-12),
arrowprops=dict(facecolor='black', shrink=0.2))
plt.show()





fig,axs = plt.subplots(3,sharex=True,sharey=True)
fig.suptitle('Signal with FFT transformation in log scale (all defect modes)')
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