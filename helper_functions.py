
'''
CONTENTS

1)X AND Y SET CREATORS

2)FOURIER SIGNAL NORMALIZATION, SIGNAL PROPERTIES AND HARMONICS EXTRACTION

3)DATA TRANSFORMATIONS

4)FEATURE ENGINEERING TECHNIQUES

5)PLOTS

6)EXPERIMENT RUN

7)TOOLS FOR TUNING
'''



########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
1) X AND Y SET CREATORS

---> x set creator (X_set)
Takes as input the path and a transformation. The outputs are five:the first is the concatenated amplitude of all three sensors
the second,third and fourth are the amplitudes of the second,third and fourth sensor, the fifth is the frequency 


---> y set creator (classification and regression) (y_set)
Takes as input the path. The output is a dataframe containing columns with elements of each sample. These elements are: damage percentage as 'dmg',
the filename as 'damage_file_name', the case study as 'caseStudey', the kind of defect as 'defect' and the index number as 'dmg_index_number'


'''

def X_set(path,transformation):

    '''
    transformations are : 'none','fourier','psd','pwelch','spectrogram','wavelet'
    
    '''
    import os
    import glob
    import numpy as np
    import pandas as pd


    sensor_data_list = []
    name_list = []

    # remove .csv from filepath so that it reads the number
    for filename in sorted(glob.glob(os.path.join(path , "data*"))):
        filename = filename.removesuffix('.csv')
        name_list.append(filename)

    #index is the number of the filename
    sensor_data = pd.DataFrame({'name':name_list})
    sensor_data['sensor_index_number'] = [int(i.split('_')[-1]) for i in sensor_data['name']]

    #list is sorted according to the index
    sensor_data = sensor_data.sort_values(by=['sensor_index_number'])

    suffix='.csv'
    new_names=[]

    #adds .csv to every filename on the list
    for filename in sensor_data['name']:
        filename = filename+suffix
        new_names.append(filename)

    #opens files and creates lists with data

    for filename in new_names:
        df = pd.read_csv(filename,sep=' |,', engine='python').dropna()
        sensor_data_list.append(df)

    freq_list = []
    power_spectrum_list = []
    sensor_names = ['s2','s3','s4']
    for sensor in sensor_names:
        for i in range(0,len(sensor_data_list)):
            sample_sensor =sensor_data_list[i][sensor]
            if transformation == 'fourier':
                power_spectrum = fourier(sample_sensor)[0]
            elif transformation == 'psd':
                power_spectrum = psd(sample_sensor)[0]
            elif transformation == 'pwelch':
                power_spectrum = pwelch(sample_sensor)[0]
            elif transformation == 'wavelet':
                power_spectrum = wavelet(sample_sensor)
            elif transformation == 'none':
                power_spectrum = sample_sensor
            elif transformation == 'spectrogram':
                power_spectrum = spectrogram(sample_sensor)
            power_spectrum_list.append(power_spectrum)  

    sensor2_vector = []
    sensor3_vector = []
    sensor4_vector = []

    bound_1 = int(len(power_spectrum_list)/3)
    bound_2 = int(2*len(power_spectrum_list)/3)
    bound_3 = int(len(power_spectrum_list))

    if transformation == 'fourier':
        for i in range(0,bound_1):
            freq_list.append(fourier(sample_sensor)[1])

    for i in range(0,bound_1):
        sensor2_vector.append(power_spectrum_list[i])
        
    for i in range(bound_1,bound_2):
        sensor3_vector.append(power_spectrum_list[i])
        
    for i in range(bound_2,bound_3):
        sensor4_vector.append(power_spectrum_list[i])
        
    X = np.concatenate((sensor2_vector,sensor3_vector,sensor4_vector),axis=1)
    return X,sensor2_vector,sensor3_vector,sensor4_vector,freq_list


def y_set(path):
    
    '''
    select column ['dmg'] which is the damage percentage for regression or ['defect'] which is the defect for classification
    
    '''
    import numpy as np
    import pandas as pd
    import os
    import glob

    dmg_list = []
    name_list = []
    case_list = []
    defect_list =[]
    for filename in glob.glob(os.path.join(path , "meta*")):
        df = pd.read_csv(filename,sep=' |,', engine='python')
        dmg_perc = df['Damage_percentage']
        case = df['caseStudey'][0]
        dmg_perc = dmg_perc[0]
        dmg_list.append(dmg_perc)
        filename = filename.removesuffix('.csv')
        
        df_defect = df['DamageLayer1'][0] + df['DamageLayer3'][0] + df['DamageLayer5'][0]
        dm_defect = df['DamageLayer1'][1] + df['DamageLayer3'][1] + df['DamageLayer5'][1]
        dd_defect = df['DamageLayer2'][0] + df['DamageLayer4'][0]
        
        if df_defect ==0 and dm_defect ==0 and dd_defect ==0:
            defect_list.append('clean')
        elif df_defect !=0 and dm_defect !=0 and dd_defect !=0:
            defect_list.append('all defect modes')
        elif df_defect !=0 and dm_defect ==0 and dd_defect ==0:
            defect_list.append('df')
        elif df_defect ==0 and dm_defect !=0 and dd_defect ==0:
            defect_list.append('dm')
        elif df_defect ==0 and dm_defect ==0 and dd_defect !=0:
            defect_list.append('dd')
        else:
            defect_list.append('all defect modes')
        
        name_list.append(filename)
        case_list.append(case)
 
    dmg_data = pd.DataFrame({'dmg':dmg_list,'damage_file_name':name_list,'caseStudey':case_list,'defect':defect_list})
    dmg_data['dmg_index_number'] = [int(i.split('_')[-1]) for i in dmg_data['damage_file_name']]
    dmg_data = dmg_data.sort_values(by=['dmg_index_number'])
    return dmg_data
    
########################################################################

########################################################################

########################################################################

########################################################################

########################################################################


'''

2) FOURIER SIGNAL NORMALIZATION, SIGNAL PROPERTIES AND HARMONICS EXTRACTION

- - - A) FOURIER SIGNAL NORMALIZATION

- - - B) SIGNAL PROPERTIES EXTRACTION WITH NORMALIZATION

- - - C) HARMONICS WITH NORMALIZATION 

'''



############################################################

'''

A) FOURIER SIGNAL NORMALIZATION


the following functions work together
to get the normalized fft use fourier_nrm_vector
which takes as input the path and the output is the normalized fft

---> fourier signal normalization (fourier_signal_normalization)
The input is a sample of signal, it calculates its fft and it then is normalized according to the excitation frequency
The amplitude is normalized according to the amplitude of the excitation frequency and the frequency is normalized according to the excitation frequency

---> fourier vector maker (fourier_vector_maker)
The input is a list of signals and the function 'fourier_signal_normalization' is applied and calculates the normalized fft.
The output is a list of normalized amplitudes and a list of normalized frequencies.

---> fourier nrm vector (fourier_nrm_vector)
The input is the path of the file containing the data. The function 'fourier_vector_maker' is applied for every sensor of every signal in that path.
The output is the concatenated normalized fft of every sensor of every signal and the normalized frequency. 

'''



def fourier_signal_normalization(sample):

    '''
    the input is a signal and the outputs are the amplitude and the frequency of the normalized fft
    
    '''
    import numpy as np
    
    amp= fourier(sample)[0]
    freq= fourier(sample)[1]

    amp_list =[]
    freq_list =[]
    bound = int(0.5*len(amp))
    max_amp = -max(amp)
    max_freq = abs(freq[amp.argmax()])

    for i in range(0,bound):
        amp_list.append(amp[i]/max_amp)
        freq_list.append(freq[i]/max_freq)

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq


def fourier_vector_maker(data):

    '''The input is the concatenated signal and the outputs are two vectors, the 
    concatenated normalized amplitude and the normalized frequency'''
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_normalization(sample)[0])
        freq_vector.append(fourier_signal_normalization(sample)[1])
    return feature_vector,freq_vector


def fourier_nrm_vector(path):
    '''
    The input is the data path and the output is the concatenated normalized fft for every sensor and the normalized frequency
    
    '''
    import numpy as np
    from helper_functions import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( fourier_vector_maker(s2)[0],fourier_vector_maker(s3)[0],fourier_vector_maker(s4)[0],fourier_vector_maker(s4)[1]),axis=1)
    return vector



############################################################

############################################################

'''

B) SIGNAL PROPERTIES EXTRACTION WITH NORMALIZATION


The following functions work together
To get the normalized fft with its properties use 'fourier_std_with_props_vector'
The input is the path and the output is the normalized fft with its properties
The properties are the excitation frequency and the reflection of the excitation frequency, their difference and the difference of their frequencies

To get the raw signal with the properties of the normalized fft use 'signal_with_props_vector' with transformation = 'none'
The input is the path and the output is the signal with the properties

To get only the properties use 'props_vector'
The input is the path and the ouptu are the properties of the normalized fft

---> signal properties extraction (signal_props_extract)
The input is a sample of normalized fft and the output is a tuple of the properties

---> signal properties extraction run (run_signal_extract)
The input is a list of signals that their fft is calculated and then are normalized after that 
for every sample the function 'signal_props_extract' is applied. The output is a list of the properties.

---> raw signal with properties (signal_with_props_vector)
The input is the path and the transformation that will be applied on the signal the output is an array of the transformed signal and the
properties of the normalized fft. It uses the function 'X_set' for the signal. The output is a concatenated array of the transformed signal and the properties
of the normalized fft.

---> properties vector (props_vector)
The input is the path and the output is an array of the concatenated properties of the normalized signal for all sensors.
For every sensor 'run_signal_extract' is applied and it calculates the properties of the signal for every sensor

---> normalized fourier signal with properties (fourier_nrm_with_props_vector)
The input is the path and the output is an array of the normalized fft and its properties
It uses the function 'X_set' for the signla which is then transformed with fft and normalized and then its properties are calculated
The output is a concatenated array of the normalized amplitude and frequency of the fft and its properties.

'''

def signal_props_extract(sample):


    '''  

    The input is a fft sample and the output is a tuple of its properties
    The bounds change depending on the material and the excitation frequency
    The default bounds are
    for the fft : freq = 0 , freq = 200 kai freq = 400 Khz
    for the normalized fft :freq = 0 , 1.3<=freq<=1.5  kai 2.9<=freq<=3.2 
    
    '''

    freq = sample[1]
    amp = sample[0]
    
    
    for i in range(0,len(freq)):

        if freq[i] >= 1.3 and freq[i] <= 1.5:
            first_bound = i
        if freq[i] >= 2.9 and freq[i] <= 3.2:
            second_bound = i
        if freq[i] ==0:
            zero_bound = i

    first_amp =[]
    for i in range(zero_bound,first_bound):
        first_amp.append(amp[i])

    second_amp =[]
    for i in range(first_bound,second_bound):
        second_amp.append(amp[i])
    
    for i in range(zero_bound,first_bound):
        if amp[i] == max(first_amp):
            first_max_amp = amp[i]
            first_max_freq = freq[i]
    

    for i in range(first_bound,second_bound):
        if amp[i] == max(second_amp):
            second_max_amp = amp[i]
            second_max_freq = freq[i]


    dx = second_max_freq-first_max_freq
    dy = first_max_amp-second_max_amp
    props = first_max_amp,second_max_amp,dx,dy
    
    return props

def run_signal_extract(data):

    '''
    The input is a list of fft samples and the output is a list of every samples' properties 
    
    '''
    feature_vector=[]
    for sample in data:
        sample = fourier_signal_normalization(sample)
        feature_vector.append(signal_props_extract(sample))
    return feature_vector

def signal_with_props_vector(path,transformation):
    

    '''
    The input is the path and the output is the transformed signal with the properties of the normalized fft

    '''

    import numpy as np
    from helper_functions import X_set
    X, s2,s3,s4,freqs = X_set(path,transformation)
    vector = np.concatenate((s2,s3,s4,freqs),axis=1)
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    prop_vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    vector = np.concatenate((vector,prop_vector),axis=1)
    return vector


def props_vector(path):

    '''
    The input is the path and the output is a tuple of the normalized fft properties

    '''
    import numpy as np
    from helper_functions import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    return vector

def fourier_nrm_with_props_vector(path):

    '''
    The input is the path and the output is the normalized fft and its properties
    
    '''
    import numpy as np
    from helper_functions import X_set
    vector = fourier_nrm_vector(path)
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    prop_vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    vector = np.concatenate((vector,prop_vector),axis=1)
    return vector
############################################################

############################################################

'''

C) HARMONICS WITH NORMALIZATION 

The following functions work together
To get the harmonics of the normalized fft use 'fourier_nrm_vector_harmonics'
That takes as input the path and the output is an array of the harmonics of every sample in that path

---> fourier harmonics (fourier_harmonics)
takes as input a signal sample and applies the fft transformation. The output is the amplitude and the frequency of the dominant harmonic.

---> harmonics from fourier signal normalization (fourier_signal_normalization_harmonics)
The input is a signal in which fft is applied and it is then normalized. Then the samples that contain the dominant harmonic are kept
and the rest is removed (These samples are 150 - 200). The output is a tuple of the normalized amplitude and frequency of the dominant harmonic. 

---> fourier harmonics vector maker (fourier_vector_maker_harmonics)
The input is an array of signal and for every signal the function 'fourier_signal_normalization_harmonics' is applied and the output is a list 
of normalized amplitudes and frequencies of the harmonic of every signal.

---> harmonics fourier nrm vector (fourier_nrm_vector_harmonics)
The input is the path and for every sample the function 'fourier_vector_maker_harmonics' is applied and the output is the concatenated array
of normalized amplitudes and frequencies of the harmonic of every sample in that path.

'''

def fourier_harmonics(sample):
    
    '''
    The input is the signal
    The output is the amplitude and the frequency of the dominant harmonic
    The dominant harmonic occurs between samples 170 and 250
    '''
    import numpy as np

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


def fourier_signal_normalization_harmonics(sample):
    
    '''
    The input is a signal
    The output is the amplitude and the frequency of the normalized fft of the dominant harmonic
    '''
    
    import numpy as np
    
    amp= fourier(sample)[0]
    freq= fourier(sample)[1]

    amp_list =[]
    freq_list =[]
    max_amp = -max(amp)
    max_freq = abs(freq[amp.argmax()])

    for i in range(150,200):
        amp_list.append(amp[i]/max_amp)
        freq_list.append(freq[i]/max_freq)

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq


def fourier_vector_maker_harmonics(data):

    '''
    The input is a list of signals
    The outputs are a list of the normalized amplitudes and a list of the normalized frequencies of the dominant harmonics
    
    '''
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_normalization_harmonics(sample)[0])
        freq_vector.append(fourier_signal_normalization_harmonics(sample)[1])
    return feature_vector,freq_vector


def fourier_nrm_vector_harmonics(path):

    '''
    The input is the path
    The output is a concatenated array of normalized amplitudes and frequencies of the harmonic of every sample in that path
    '''
    
    import numpy as np
    from helper_functions import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( fourier_vector_maker_harmonics(s2)[0],fourier_vector_maker_harmonics(s3)[0],fourier_vector_maker_harmonics(s4)[0],fourier_vector_maker_harmonics(s4)[1]),axis=1)
    return vector



############################################################


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
3) DATA TRANSFORMATIONS

---> fast fourier transform (fourier)

The input is a signal and the output is the amplitude and the frequency of the fft of the signal

---> pwelch (pwelch)
The input is a signal and the output is the amplitude and the frequency of the pwelch of the signal

---> psd (psd)
The input is a signal and the output is the amplitude and the frequency of the psd of the signal

---> spectrogram (spectrogram)
The input is a signal and the output is the spectrogram signal

---> wavelet (wavelet)
The input is a signal and the output is the db1 wavelet of the signal

---> noise adder (add_noiz)
The input is the output of the 'X_set' function and the output is the output of X_set with noise added
'''

def fourier(sample_sensor):
    '''
    The input is a signal 
    The output is the amplitude and the frequency of the fft of the signal
    '''
    import numpy as np
    fs = 1/1000
    fourier = np.fft.fft(sample_sensor)
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    power_spectrum = np.log(power_spectrum)
    
    return power_spectrum,freqs


def pwelch(sample_sensor):

    '''
    The input is a signal 
    The output is the amplitude and the frequency of the pwelch of the signal
    '''

    from scipy import signal
    fs = 1000
    (f, S)= signal.welch(sample_sensor, fs, nperseg=1024)
    return S,f
    #plt.semilogy(f, S)
    #plt.xlim([0, 500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def psd(sample_sensor):

    '''
    The input is a signal 
    The output is the amplitude and the frequency of the psd of the signal
    '''

    from scipy import signal
    fs = 1000
    (f, S) = signal.periodogram(sample_sensor, fs, scaling='density')
    return S,f
    #plt.semilogy(f, S)
    #plt.ylim([1e-14, 1e-3])
    #plt.xlim([0,500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def spectrogram(sample):

    '''
    The input is a signal 
    The output is the spectrogram of the signal
    '''

    from scipy import signal

    fs = 1000
    f, t, Sxx = signal.spectrogram(sample, fs)
    #plt.pcolormesh(t, f, Sxx, shading='gouraud')
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
    return Sxx

def wavelet(sample):
    '''
    The input is a signal 
    The output is the db1 wavelet of the signal
    '''
    import pywt
    import numpy as np

    fs = 1000  
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = sample

    wavelet_name = 'db1' 
    transformed_signal, _ = pywt.dwt(signal, wavelet_name)
    return transformed_signal
    # Plot the original signal
    #plt.subplot(2, 1, 1)
    #plt.plot(signal)
    #plt.title('Original Signal')

    # Plot the transformed signal
    #plt.subplot(2, 1, 2)
    #plt.plot(transformed_signal)
    #plt.title('Transformed Signal')

    #plt.tight_layout()
    #plt.show()

def add_noiz(X_set,mean,stdev):

    '''
    The input is the output of the 'X_set' function 
    The output is the output of X_set with noise added
    '''

    import numpy as np
    X_set_new =[]
    for sample in X_set:
        noise = np.random.normal(mean,stdev, len(sample))
        sample = sample+noise
        X_set_new.append(sample)

    X_set = X_set_new
    return X_set

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################


'''
4) FEATURE ENGINEERING TECHNIQUES

---> prinicipal component analysis (pca)
The input is the X_train and X_test and this function performs pca. PCA finds linear combinations with the largest variance and creates principal 
components with these combinations. The results are the X_train and X_test that instead of the original data they contain the principal components.

---> kernel principal component analysis (kpca)
The input is the X_train and X_test and this function performs kernel pca. First the data are projected in a feature space according to the choice of the kernel. 
Then PCA finds linear combinations with the largest variance and creates principal components with these combinations. The results are the X_train and X_test that 
instead of the original data they contain the principal components from the feature space they were projected.

---> data mixer (data_mixer)
The input is the X and Y data of one dataset and those of another dataset and the percentage that is kept from each dataset
Then the percentages of each dataset are combined and shuffled
The output is the shuffled combination of the two datasets.
'''


def pca(X_train,X_test):

    '''
    The input is the original X_train and X_test
    The output is the X_train and X_test that contain the principal components of the original data
    '''
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    pca = PCA(n_components=50, random_state = 42)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = pca.transform(X_test)
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    #plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    #plt.ylabel('Percentage of Explained Variance')
    #plt.xlabel('Principal Component')
    #plt.title('Scree Plot')
    #plt.show()
    return X_train,X_test


def kpca(X_train,X_test,input_kernel):


    '''
    The input is the original X_train and X_test and the kernel
    The output is the X_train and X_test that contain the principal components of the original data that are projected to a feature space 
    The feature space depends on the chosen kernel

    The kernels are :
    Periodic
    Locally periodic
    RBF
    Rational quadratic
    Rational locally periodic
    '''
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import KernelPCA   
    from sklearn.gaussian_process.kernels import ExpSineSquared,Product,RationalQuadratic,RBF

    periodic= ExpSineSquared()
    locally_periodic = Product(periodic,RBF())
    rational_locally_periodic = Product(periodic,RationalQuadratic())


    if input_kernel =='periodic':
        input_kernel = periodic
    if input_kernel =='locally_periodic':
        input_kernel = locally_periodic
    if input_kernel =='rbf':
        input_kernel = RBF()
    if input_kernel =='rational_quadratic':
        input_kernel = RationalQuadratic()
    if input_kernel =='rational_locally_periodic':
        input_kernel = rational_locally_periodic
    
    pca = KernelPCA(kernel=input_kernel,n_components=30, random_state = 42)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = pca.transform(X_test)
    X_test = pd.DataFrame(X_test)
    return X_train,X_test

def data_mixer(X_1,y_1,X_2,y_2,first_percentage,second_percentage):
    '''
    
    The input is the X and Y data of two datasets each dataset's percentage that is kept
    The output is the shuffled combination of the two datasets.
    
    '''
    from sklearn.model_selection import train_test_split
    import numpy as np
    if first_percentage == 1:
        X_1_half = X_1
        y_1_half = y_1        
    else:
        X_1_half, X_drop, y_1_half, y_drop = train_test_split(X_1, y_1, test_size=1-first_percentage,shuffle=True)
    
    if second_percentage ==1:
        X_2_half = X_2
        y_2_half = y_2
    else:
        X_2_half, X_drop, y_2_half, y_drop = train_test_split(X_2, y_2, test_size=1-second_percentage,shuffle=True)
    
    X_train = np.concatenate((X_1_half,X_2_half),axis=0)
    y_train = np.concatenate((y_1_half,y_2_half),axis=0)
    return X_train,y_train


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
5) PLOTS



---> figures with subplots of all damage scenarios for every sensor(all_damage_every_sensor_separate)
takes as input the data path and a list that contains tuples which contain the indexes of each sample, the damage 
percentage values or defects, the color of the line, and the linestyle. It plots three figures, one for every sensor,
each figure contains subplots of all damage scenarios of the given samples.

---> figure of all damage scenarios for every sensor(all_damage_every_sensor_together)
takes as input the data path and a list that contains tuples which contain the indexes of each sample, the damage 
percentage values or defects, the color of the line, and the linestyle. It plots three figures, one for every sensor,
each figure contains plots of all damage scenarios of the given samples.

---> signal plot of every sensor (all_sensor_time_plot_separate)
takes as input the data path and the index of the sample that is plotted. Plots three subplots of the 
time signal of every sensor for the sample of that index.

---> fft plot of every sensor (all_sensor_fft_plot_separate)
takes as input the data path and the index of the sample that is plotted. Plots three subplots of the 
fft of every sensor for the sample of that index.

---> fft plot of a single sensor (single_sensor_fft_plot)
takes as input the data path the index of the sample that is plotted and the name of the defect and plots the fft of sensor 3. 
Two arrows show the excitation frequency and the dominant harmonic frequency.

---> harmonics plot for every defect mode (every_defect_mode_harmonics_plot)
takes as input the data path and the sample indexes for every kind of defect(dd,df,dm,all) and plots 4
subplots of the harmonics of every defect mode for one sensor

---> parity plots it can either save or show the plot (parity_plot)
pairnei san input to y_test to y_pred to montelo kai to mode dhladh an thelw na kanw save h aplws na dw to plot
bgazei to parity plot tou y_test me to y_pred kai eite to kanei save eite to deixnei


---> confusion matrix gia to classification task (confusion_matrix_display)
pairnei san input ta y_test,y_pred,model,mode,accuracy kai bgazei to confusion matrix me titlo
to onoma tou montelou kai to accuracy tou. To montelo prepei na einai function kai to mode einai 
eite show eite save.


---> regression results bar chart (regression_results_bar_charts)
the inputs are the model names, the mape values, the standard deviation values, the p-value values as lists and the label on y axis.

---> classification results bar chart (classification_results_bar_charts)
the inputs are the model names, the mape values, the standard deviation values, the p-value values as lists and the label on y axis.

'''


def all_damage_every_sensor_separate(path,index_list):
    
    '''
    takes as input the data path and a list that contains tuples which contain the indexes of each sample, the damage 
    percentage values or defects, the color of the line, and the linestyle
    The output are three figures, one for every sensor,
    each figure contains subplots of all damage scenarios of the given samples.
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np
    from helper_functions import fourier, X_set

    plt.rcParams.update({'font.size': 12})
    plt.rcParams['lines.linewidth'] = 2.5

    __, s2, s3, s4, __ = X_set(path, 'none')
    sensor_list = [(s2, 'sensor 2'), (s3, 'sensor 3'), (s4, 'sensor 4')]

    for sensor_data, sensor_name in sensor_list:
        num_plots = len(index_list)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 2.8 * num_plots), sharex=True)

        if num_plots == 1:
            axs = [axs]

        for ax, (idx, label, color, linestyle) in zip(axs, index_list):
            fft_amplitude, fft_freq = fourier(sensor_data[idx])
            ax.plot(np.abs(fft_freq), fft_amplitude,
                    color=color, linestyle=linestyle, label=label)
            ax.grid(True)
            ax.legend(loc='upper right')

        axs[-1].set_xlabel('Frequency (kHz)')

        fig.text(0.04, 0.5, 'Log Amplitude (V)', va='center', rotation='vertical', fontsize=16)
        fig.suptitle(f'FFT plots of {sensor_name}', fontsize=18)
        fig.tight_layout(rect=[0.06, 0, 1, 0.95])
        plt.show()


def all_damage_every_sensor_together(path,index_list):
    
    '''
    The input is the data path and a list that contains tuples which contain the indexes of each sample, the damage 
    percentage values or defects, the color of the line, and the linestyle.
    The outputs are three figures, one for every sensor,
    each figure contains plots of all damage scenarios of the given samples.
    '''

    import matplotlib.pyplot as plt
    import numpy as np
    from helper_functions import fourier, X_set

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5

    __, s2, s3, s4, __ = X_set(path, 'none')
    sensor_list = [(s2, 'sensor 2'), (s3, 'sensor 3'), (s4, 'sensor 4')]

    for sensor_data, sensor_name in sensor_list:
        plt.figure()
        legend_list = []
        y_offset = 0 

        for idx, damage, linecolor, linestyle in index_list:
            fft_amplitude, fft_freq = fourier(sensor_data[idx])
            
            amplitude_with_offset = fft_amplitude + y_offset
            plt.plot(np.abs(fft_freq), amplitude_with_offset, color=linecolor, linestyle=linestyle)
            legend_list.append(f"{damage} (offset {y_offset})")
            y_offset += 2.5

        plt.title(f'FFT in log scale of {sensor_name}')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Log Amplitude + Offset')
        plt.legend(legend_list)
        plt.grid(True)
        plt.show()

def all_sensor_time_plot_separate(path,index):

    '''
    The input is the data path and the index of the sample that is plotted. 
    The outputs are three subplots of the time signal of every sensor for the sample of that index.
    '''
    
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


def all_sensor_fft_plot_separate(path,index):


    '''
    The input is the data path and the index of the sample that is plotted.
    The outputs are three subplots of the fft of every sensor for the sample of that index
    '''
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


def single_sensor_fft_plot(path,index,defect):
    
    '''
    The input is the data path, the index of the sample that is plotted and the name of the defect 
    The output is a plot of the fft of sensor 3. 
    Two arrows show the excitation frequency and the dominant harmonic frequency.
    
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    __,s2,s3,s4,__ = X_set(path,'none')

    s2 = fourier(s2[index])
    s3 = fourier(s3[index])
    s4 = fourier(s4[index])

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    
    plt.title(f'Signal with FFT transformation in log scale of sensor 3 ({defect})')
    plt.xlabel( 'Frequency (kHz)')
    plt.ylabel('Amplitude (v)')
    plt.plot(np.abs(s3[1]),s3[0])
    plt.grid()
    plt.annotate('Excitation frequency (125kHz)', xy=(125,-4), xytext=(200, -3),
    arrowprops=dict(facecolor='black', shrink=0.2))
    plt.annotate('Dominant harmonic (250 kHz)', xy=(250,-10), xytext=(120,-12),
    arrowprops=dict(facecolor='black', shrink=0.2))
    plt.show()

def every_defect_mode_harmonics_plot(path,dd_index,df_index,all_index,dm_index):

    '''
    The input is the data path and the sample indexes for every kind of defect(dd,df,dm,all) 
    The outputs are 4 subplots of the harmonics of every defect mode for one sensor

    to see all defects:
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



def parity_plot(y_test,y_pred,model,mode):
    
    
    '''
    The inputs are y_test, y_pred the model and the mode 
    In 'save' mode the plot is saved and in 'show' mode the plot is shown
    The output is the parity plot of y_test and y_pred 

    the model has to be a function

    '''

    import matplotlib.pyplot as plt

    plt.scatter(y_test,y_pred,color='r')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints)
    plt.xlabel('Test Values')
    plt.ylabel('Predicted Values')
    if model.__name__ =='mlp' : name = 'MLP'
    if model.__name__ =='linear_regression' : name = 'Linear Regression'
    if model.__name__ =='decision_tree_reg' : name = 'Decision Trees'
    if model.__name__ =='cnn_reg' : name = 'CNN'
    plt.title(f'Parity plot of {name}')
    plt.legend(["y_values", "y=x"], loc="lower right")
    if mode=='save':
        plt.savefig(f'{name}_parity_plot.png')
        plt.close('all')
        plt.clf()
    elif mode =='show':
        plt.show()


def confusion_matrix_display(y_test,y_pred,model,mode,accuracy):
    
    '''
    The inputs are y_test,y_pred,model,mode, the accuracy of the predictions 
    In 'save' mode the plot is saved and in 'show' mode the plot is shown
    The output is the confusion matrix and its title says the model's name and accuracy
    
    The model has to be a function
    '''
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

   
    if model.__name__ =='svc' : name = 'Support Vector Machines'
    if model.__name__ =='random_forest_clf' : name = 'Random Forest'
    if model.__name__ =='xgb_clf' : name = 'XGB'
    if model.__name__ =='cnn_class' : name = 'CNN'
    cm = confusion_matrix(y_test,y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f'Confusion matrix of {name} with accuracy = {accuracy}')
    if mode=='save':
        plt.savefig(f'{name}_confusion_matrix.png')
        plt.close('all')
        plt.clf()
    elif mode =='show':
        plt.show()


def regression_results_bar_charts(model_names, mape, std_devs, pvals,ylabel):
    

    '''
    The inputs are the model names, the mape values, the standard deviation values, 
    the p-value values as lists and the label on y axis.
    The output is a figure that contains bar charts that compare each models 
    mape, standard deviation and p-value
    '''
    
    
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    
    mape = np.array(mape)
    std_devs = np.array(std_devs)
    pvals = np.array(pvals)
    x = np.arange(len(model_names))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, mape, capsize=5, color='skyblue', edgecolor='black')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f'{mape[i]:.6f} ± {std_devs[i]:.6f}\n(p-value: {pvals[i]:.6f})' # bazw eite to F1: eite to p-value:
        plt.text(bar.get_x() + bar.get_width()/2, height + 0, label, # dipla sto height bazw +/- kapoion arithmo gia na einai pio omorfo
                 ha='center', va='bottom', fontsize=16)

    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title('Model performance comparison')
    plt.ylim(0, max(mape + std_devs) -0.02) # sto telo bazw +/- enan arithmo gia na einai pio omorfo to chart
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def classification_results_bar_charts(model_names, accuracies, std_devs, f1_scores,ylabel):
    
    '''
    
    The inputs are the model names, the accuracy values, the standard deviation values, the f1 scores as lists and the label on y axis.
    The output is a figure with bar charts that compare each model's accuracy standard deviation and f1 score
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5

    accuracies = np.array(accuracies)
    std_devs = np.array(std_devs)
    f1_scores = np.array(f1_scores)

    x = np.arange(len(model_names))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, accuracies, capsize=5, color='skyblue', edgecolor='black')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f'{accuracies[i]:.4f} ± {std_devs[i]:.4f}\n(F1: {f1_scores[i]:.4f})'
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, label,
                 ha='center', va='bottom', fontsize=16)

    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title('Model performance comparison')
    plt.ylim(0, max(accuracies + std_devs) + 0.1)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################



'''

6)EXPERIMENT RUN

---> regression experiment run (regression_model_run)
The input is the model as a function the X_train, y_train, X_test, y_test. That model is trained and tested and mae and mape are calculated.
The outputs are mae,mape,y_test and y_pred 

---> classification experiment run(classification_model_run)
The input is the model as a function the X_train, y_train, X_test, y_test. That model is trained and tested and its accuracy is calculated.
The outputs are accuracy,y_test kai y_pred
'''


def regression_model_run(model,X_train,y,X_test,y_test):
    
    '''
    The input is the model as a function the X_train, y_train, X_test, y_test.
    The outputs are mae,mape,y_test and y_pred of that model
    '''
    from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error

    y_pred = model(X_train,y,X_test)
    mape = 100*mean_absolute_percentage_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    return mae,mape,y_test,y_pred

def classification_model_run(model,X_train,y,X_test,y_test):

    '''
    The input is the model as a function the X_train, y_train, X_test, y_test.
    The outputs are accuracy,y_test and y_pred of that model
    '''

    from sklearn.metrics import accuracy_score
    y_pred = model(X_train,y,X_test)
    acc = 100*accuracy_score(y_test,y_pred)
    acc = accuracy_score(y_test,y_pred)
    return acc,y_test,y_pred

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################



'''

7)TOOLS FOR TUNING

---> cross validation me leave one out(cross_val_loo)

The input is the model, the X and y data. This function performs leave one out cross validation and calculates the scores
for every fold. For regression the scoring is 'neg_mean_absolute_percentage_error'  and for classification the scoring is 'accuracy'
The output is the score of each fold


---> grid search me leave one out(grid_search_loo)
The input is the model the X_train and y_train. This function performs grid search with leave one out to find the best parameters for the model according to a scoring.
Depending on the model the parameters defer.
The outputs are the parameters with which the model achieved its best performance
'''

def cross_val_loo(model,X,y):
    '''
    The input is the model, the X and y data.
    The output is the score of each fold after cross validation with leave one out
    
    
    regression --> scoring = 'neg_mean_absolute_percentage_error'
    classification ---> scoring = 'accuracy'
    '''
    from sklearn.model_selection import LeaveOneOut,cross_val_score
    import numpy as np
    cv = LeaveOneOut()
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return np.absolute(scores)



def grid_search_loo(model,X_train,y_train):
    '''
    The input is the model the X_train and y_train
    The outputs are the parameters with which the model achieved its best performance after leave one out grid search

    Each model has different parameters, the default parameters are the parameters of an SVM algorithm
    
    '''
    from sklearn.model_selection import GridSearchCV,LeaveOneOut
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['rbf','sigmoid','poly']} 

    grid = GridSearchCV(model, param_grid, refit = True, cv=LeaveOneOut(),verbose = False) 
    grid.fit(X_train, y_train) 

    print(grid.best_params_)
    print(grid.best_score_)


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################