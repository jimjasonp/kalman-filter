from helper_functions import X_set,fourier,y_set


def all_sensor_time_plot_separate(path,index):
    import matplotlib.pyplot as plt

    __,s2,s3,s4,__ = X_set(path,'none')

    s2 = s2[index]
    s3 = s3[index]
    s4 = s4[index]

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    
    fig,axs = plt.subplots(3,sharex=True,sharey=True)
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

    plt.show()

def all_sensor_fft_plot_together(path,index):
    import matplotlib.pyplot as plt
    import numpy as np
    __,s2,s3,s4,__ = X_set(path,'none')

 
    s2 = fourier(s2[index])
    s3 = fourier(s3[index])
    s4 = fourier(s4[index])

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    
    plt.plot(np.abs(s2[1]),s2[0],linestyle ='dashed', dashes = (10,15))
    plt.plot(np.abs(s3[1]),s3[0],linestyle ='-.')
    plt.plot(np.abs(s4[1]),s4[0],linestyle ='solid')
    plt.grid()
    plt.title('Signal with FFT transformation in log scale (all defect modes)')
    plt.ylabel('Amplitude (v)')
    plt.xlabel('Frequency (kHz)')
    plt.legend(['s2','s3','s4'])
    plt.show()


def single_sensor_fft_plot(path,index):
    import matplotlib.pyplot as plt
    import numpy as np

    '''
    gia na fainetai omorfo bale path = 'random_data' kai index 10 
    
    '''
    __,s2,s3,s4,__ = X_set(path,'none')

    s2 = fourier(s2[index])
    s3 = fourier(s3[index])
    s4 = fourier(s4[index])

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    
    plt.title('Signal with FFT transformation in log scale of sensor 3 (all defect modes)')
    plt.xlabel( 'Frequency (kHz)')
    plt.ylabel('Amplitude (v)')
    plt.plot(np.abs(s3[1]),s3[0])
    plt.grid()
    plt.annotate('Excitation frequency (125kHz)', xy=(125,-4), xytext=(200, -3),
    arrowprops=dict(facecolor='black', shrink=0.2))
    plt.annotate('Dominant harmonic (250 kHz)', xy=(250,-10), xytext=(120,-12),
    arrowprops=dict(facecolor='black', shrink=0.2))
    plt.show()




def all_sensor_fft_plot_separate(path,index):
    import matplotlib.pyplot as plt
    import numpy as np

    __,s2,s3,s4,__ = X_set(path,'none')

    s2 = fourier(s2[index])
    s3 = fourier(s3[index])
    s4 = fourier(s4[index])

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5

    fig,axs = plt.subplots(3,sharex=True,sharey=True)
    fig.suptitle('Signal with FFT transformation in log scale (all defect modes)')
    fig.text(0.5 , 0.04, 'Frequency (kHz)',ha = 'center')
    fig.text(0.04 , 0.5, 'Amplitude (v)',va = 'center',rotation = 'vertical')

    axs[0].plot(np.abs(s2[1]),s2[0])
    axs[0].grid()
    axs[0].set_title('s2')

    axs[1].plot(np.abs(s3[1]),s3[0])
    axs[1].grid()
    axs[1].set_title('s3')

    axs[2].plot(np.abs(s4[1]),s4[0])
    axs[2].grid()
    axs[2].set_title('s4')

    plt.show()



def fourier_harmonics(sample):
    
    import numpy as np
    import matplotlib.pyplot as plt

    amp= fourier(sample)[0]
    freq= np.abs(fourier(sample)[1])
    amp_list =[]
    freq_list =[]

    for i in range(170,250):
        amp_list.append(amp[i])
        freq_list.append(freq[i])

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq

def single_defect_mode_harmonics_plot(index):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    
    
    __,s2,s3,s4,__ = X_set(path,'none')
    s2 = fourier_harmonics(s2[index])
    s3= fourier_harmonics(s3[index])
    s4= fourier_harmonics(s4[index])
  
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




def every_defect_mode_harmonics_plot(path,dd_index,df_index,all_index,dm_index):

    '''
    gia na bgei vraiao ...
    path = 'Balanced_data'

    dd --> 0
    df --> 1
    all --> 54
    dm --> 58
    '''

    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    __,s2,s3,s4,__ = X_set(path,'none')

    dd = fourier_harmonics(s3[dd_index])
    df = fourier_harmonics(s3[df_index])
    all = fourier_harmonics(s3[all_index])
    dm = fourier_harmonics(s3[dm_index])
    fig,axs = plt.subplots(4,sharex=True,sharey=True)


    fig.suptitle(f'Harmonics in log scale of sensor 3')
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



path = 'random_data'


all_sensor_time_plot_separate(path,5)
all_sensor_fft_plot_separate(path,5)
all_sensor_fft_plot_together(path,60)
single_sensor_fft_plot(path,10)

path = 'Balanced_data'


'''fft_log_plot(0)
fft_log_plot(1)
fft_log_plot(54)
fft_log_plot(58)'''

every_defect_mode_harmonics_plot(path,0,1,54,58,)
single_defect_mode_harmonics_plot(0)