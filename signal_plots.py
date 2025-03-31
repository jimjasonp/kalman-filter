import matplotlib.pyplot as plt
import numpy as np
path = 'test_classification'

from helper_functions import X_set

__,s2,s3,s4,__ = X_set(path,'none')

s2 = s2[10]
s3 = s3[10]
s4 = s4[10]

'''plt.plot(s2)
plt.plot(s3)
plt.plot(s4)
plt.grid()
plt.title('Signal from all sensors')
plt.ylabel('Electric potential (v)')
plt.xlabel('time (ms)')
plt.legend(['s2','s3','s4'])
plt.show()'''

def fft(sample_sensor):
    fs = 1/500
    fourier = np.fft.fft(sample_sensor)
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    power_spectrum = np.log(power_spectrum)
    return power_spectrum,freqs

s2 = fft(s2)
s3 = fft(s3)
s4 = fft(s4)

plt.scatter(s2[1],s2[0])
plt.scatter(s3[1],s3[0])
plt.scatter(s4[1],s4[0])
plt.grid()
plt.title('Signal with FFT transformation in log scale')
plt.ylabel('Amplitude (v)')
plt.xlabel('Frequency (Hz)')
plt.legend(['s2','s3','s4'])
plt.show()