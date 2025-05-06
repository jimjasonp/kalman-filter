
'''
CONTENTS

1)X AND Y SET CREATORS

2)FOURIER SIGNAL NORMALIZATION, SIGNAL PROPERTIES AND HARMONICS EXTRACTION

3)DATA TRANSFORMATIONS

4)FEATURE ENGINEERING TECHNIQUES

5)PLOTS

6)KERNELS

7)EXPERIMENT RUN

8)TOOLS FOR TUNING
'''



########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
1) X AND Y SET CREATORS

---> x set creator (X_set)

to x set pairnei san input path kai to eidos tou transformation
bgazei 5 outputs
to prwto output einai to concatenated amplitude ten shmatwn kai twv triwn sensors
to deutero trito kai tetarto output einai to amplitude tou shmatos tou kathe sensor
to pempto output einai h suxnothta 

---> y set creator (classification and regression) (y_set)

to y_set pairnei san input to path kai bgazei san output to dataframe me ola ta stoixeia gia 
thn astoxia to column 'defect' exei to eidos tou defecr kai to column 'dmg' exei to damage percentage

'''

def X_set(path,transformation):

    import os
    import glob
    import numpy as np
    import pandas as pd


    sensor_data_list = []
    name_list = []

    # gia kathe filename sto path pou tou exw dwsei afairei to .csv wste meta na mporei na diabasei ton arithmo
    for filename in sorted(glob.glob(os.path.join(path , "data*"))):
        filename = filename.removesuffix('.csv')
        name_list.append(filename)

    #apo kathe filename krataei mono ton arithmo sto telos kai me auton ton arithmo ftiaxeni th nea sthlh index number
    sensor_data = pd.DataFrame({'name':name_list})
    sensor_data['sensor_index_number'] = [int(i.split('_')[-1]) for i in sensor_data['name']]

    #kanw sort th lista basei tou index number
    sensor_data = sensor_data.sort_values(by=['sensor_index_number'])

    suffix='.csv'
    new_names=[]

    #se kathe filename sth lista pou exei ginei sort prosthetei to .csv wste na mporei na to diabasei
    for filename in sensor_data['name']:
        filename = filename+suffix
        new_names.append(filename)

    #anoigei ta arxeia apo kathe path kai ftiaxnei th lista me tis metrhseis

    for filename in new_names:
        df = pd.read_csv(filename,sep=' |,', engine='python').dropna()
        sensor_data_list.append(df)

    freq_list = []
    power_spectrum_list = []
    sensor_names = ['s2','s3','s4']
    for sensor in sensor_names:
        #gia kathe sample sensora dld gia kathe xronoseira (pou prokuptei apo to shma pou lambanei o sensoras efarmozo transformations
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
    
    import numpy as np
    import pandas as pd
    import os
    import glob

    #### paizei mono gia to balanced data###
    dmg_list = []
    name_list = []
    case_list = []
    defect_list =[]
    # gia kathe file name sto path pou exw dwsei afairei to .csv kai afairei nan values kai kanei mia lista mono me to damage percentage
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

    # ftiaxnei ena dataframe me to damage percentage kai prosthetei to index number kai kanei sort basei autou 
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

ta tria parakatw functions leitourgoun mazi
gia na parw to kanonikopoihmeno shma xrhsimopoiw to fourier nrm vector
to opoio pairnei san input to path kai to output einai to kanonikopoihmeno shma

---> fourier signal normalization (fourier_signal_normalization)
To input einai ena sample shmatos kai ypologizei to fft kai kanonikopoei ws pros th megisth syxnothta
dhladh th syxnothta diegershs. To amplitude einai kanonikopoihmeno ws pros to amplitude sth megisth syxnothta kai 
h suxnothta einai kanonikopoihmenh ws pros th syxnothta diegershs

---> fourier vector maker (fourier_vector_maker)
Pairnei san input data (mia lista h array apo shmata) kai efarmozei th sunarthsh fourier_signal_normalization kai dinei to kanonikopoihmeno shma
to input einai ta data kai ta output einai mia lista me to kanonikopoihmeno amplitude kai mia lista me thn kanonikopoihmenh syxnothta

---> fourier nrm vector (fourier_nrm_vector)
To input einai to path kai efarmozei thn sunarthsh fourier_vector_maker gia olous tous sensors kathe shmatos sto path.
To input einai to path kai to output einai ta concatenated normalized shmata olwn twn samples sto path

'''



def fourier_signal_normalization(sample):
    import numpy as np
    

    ########## pairnei san input sample apo raw shma
    ####### dinei output to amp kai to freq tou kanonikopoihmenou shmatos
    amp= fourier(sample)[0]
    freq= fourier(sample)[1]

    amp_list =[]
    freq_list =[]
    bound = int(0.5*len(amp))
    #max_amp = max(amp)
    max_amp = -max(amp)
    max_freq = abs(freq[amp.argmax()])

    for i in range(0,bound):
        amp_list.append(amp[i]/max_amp)
        #amp_list.append(1/(amp[i]/max_amp))
        freq_list.append(freq[i]/max_freq)

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq


def fourier_vector_maker(data):
    ########## pairnei san input data
    ####### dinei output vector kanonikopoihmeno sample me freq
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_normalization(sample)[0])
        freq_vector.append(fourier_signal_normalization(sample)[1])
    return feature_vector,freq_vector


def fourier_nrm_vector(path):

    import numpy as np
    
    ########## pairnei san input path
    ####### dinei output to nrm fourier shma
    from helper_functions import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( fourier_vector_maker(s2)[0],fourier_vector_maker(s3)[0],fourier_vector_maker(s4)[0],fourier_vector_maker(s4)[1]),axis=1)
    return vector



############################################################

############################################################

'''

B) SIGNAL PROPERTIES EXTRACTION WITH NORMALIZATION

ta parakatw functions leitourgoun mazi
gia na parw to kanonikopoihmeno shma me ta props xrhsimopoiw to fourier_std_with_props_vector
to opoio pairnei san input to path kai to output einai to kanonikopoihmeno shma me ta props

gia na parw to raw shma me ta props xrhsimopoiw to signal_with_props_vector
to opoio pairnei san input to path kai to output einai to raw shma me ta props

gia na parw ta props xrhsimopoiw to props_vector
to opoio pairnei san input to path kai to output einai ta props


---> signal properties extraction (signal_props_extract)
To input einai to sample tou kanonikopoihmenou shmatos kai to output einai kapoia properties tou shmatos.
Ta properties poy ypologizei einai h diafora metaksy twn duo megalyterwn syxnothtwn kai h diafora metaksy twn duo megaluterwn amplitudes

---> signal properties extraction run (run_signal_extract)
To input einai data (lista h array me shmata) kai prwta kanonikopoiei ta shmata kai meta trexei 
gia kathe sample to signal_props_extract kai to output einai mia lista me ta properties tou kathe 
kanonikopoihmenou shmatos

---> raw signal with properties (signal_with_props_vector)
To input einai to path kai to transformation pou thelw kai to output einai ena array me to shma kai ta properties tou kanonikopoihmenoy shmatos
xrhsimopoei to function X_set gia na ftiaksei to shma kai na kanei to transformation pou tha dothei kai meta kanonikopoei to shma kai dinei ta properties tou
To output einai ena concatenated array me to original h to transformed shma me th suxnothta kai ta properties tou kanonikopoihmenou shmatos 

---> properties vector (props_vector)
To input einai to path kai output einai ena array me ta properties tou kanonikopoihmenoy shmatos
gia kathe sensora trexei run_signal_extract kai bgazei ta properties tou shmatos kathe sensor
To output einai ena array me ta concatenated signal properties kath sensor

---> normalized fourier signal with properties (fourier_nrm_with_props_vector)
To input einai to path kai to output einai ena array me to kanonikopoihmeno shma kai ta properties tou kanonikopoihmenoy shmatos
xrhsimopoei to function X_set gia na ftiaksei to shma kai meta kanonikopoei to shma kai dinei ta properties tou
To output einai ena concatenated array me to kanonikopoihmeno shma kai me thn kanonikopoihmenh suxnothta kai ta properties tou kanonikopoihmenou shmatos

'''

def signal_props_extract(sample):
    ########## pairnei san input sample apo fourier shma
    ####### dinei output ta signal props tou shmatos
    freq = sample[1]
    amp = sample[0]
    
    #### auta ta bounds allazoun analoga me ta shmeia poy emfanizetai to megisto amp
    ### gia kanoniko shma ta oria einai freq = 0 , freq = 200 kai freq = 400 Khz
    ### gia normalized shma einia freq = 0 , 1.3<=freq<=1.5  kai 2.9<=freq<=3.2 
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
            #####
            first_max_i = i
            #####
    

    for i in range(first_bound,second_bound):
        if amp[i] == max(second_amp):
            second_max_amp = amp[i]
            second_max_freq = freq[i]
            #####
            second_max_i = i
            ######

    #####
    #for i in range(zero_bound,first_max_i):
    #    if amp[i] < 0.1 * max(first_amp):
    ##        first_width_first_bound = freq[i]
    #for i in range(first_max_i,first_bound):
    #    if amp[i] < 0.1 * max(first_amp):
    #        first_width_second_bound = freq[i]

    #for i in range(first_bound,second_max_i):
    #    if amp[i] < 0.1 * max(second_amp):
    #        second_width_first_bound = freq[i]
    #for i in range(second_max_i,second_bound):
    #    if amp[i] < 0.1 * max(second_amp):
    #        second_width_second_bound = freq[i]

    #big_width = first_width_second_bound-first_width_first_bound
    #small_width = second_width_second_bound- second_width_first_bound
    #####

    dx = second_max_freq-first_max_freq
    dy = first_max_amp-second_max_amp
    logos_freq = second_max_freq/first_max_freq
    logos_amp = first_max_amp/second_max_amp
    props = first_max_amp,second_max_amp,dx,dy#,logos_freq,logos_amp#,big_width,small_width,first_max_freq,second_max_freq
    
    return props

def run_signal_extract(data):
    ########## pairnei san input raw shma
    ####### dinei output ta signal properties tou shmatos
    feature_vector=[]
    for sample in data:
        sample = fourier_signal_normalization(sample)
        feature_vector.append(signal_props_extract(sample))
    return feature_vector

def signal_with_props_vector(path,transformation):
    
    import numpy as np
    
    ########## pairnei san input path
    ####### dinei output ta signal properties tou shmatos me to shma me ton metasxhmatismo
    from helper_functions import X_set
    X, s2,s3,s4,freqs = X_set(path,transformation)
    vector = np.concatenate((s2,s3,s4,freqs),axis=1)
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    prop_vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    vector = np.concatenate((vector,prop_vector),axis=1)
    return vector


def props_vector(path):

    import numpy as np
    
    
    ########## pairnei san input path
    ####### dinei output ta signal properties tou shmatos
    from helper_functions import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    return vector

def fourier_nrm_with_props_vector(path):

    import numpy as np
    
    ########## pairnei san input path
    ####### dinei output ta signal properties tou shmatos me to shma me to kanonikopoihmeno fourier
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


---> fourier harmonics (fourier_harmonics)
takes as input a signal sample and applies the fft transformation. The output is the amplitude and the frequency of the dominant harmonic.


ta tria parakatw functions leitourgoun mazi
gia na parw tis armonikes apo to kanonikopoihmeno shma xrhsimopoiw to fourier nrm vector
to opoio pairnei san input to path kai to output einai oi armonikes tou kanonikopoihmenou shmatos

---> harmonics from fourier signal normalization (fourier_signal_normalization_harmonics)
To input einai ena sample shmatos kai ypologizei to fft kai kanonikopoei ws pros th megisth syxnothta
dhladh th syxnothta diegershs. Apo auto to shma krataei mono tis ta samples me arithmo sample apo 150 ews 200
giati mesa se auto to diasthma exw tis dominant armonikes. To amplitude einai kanonikopoihmeno ws pros to amplitude sth megisth syxnothta kai 
h suxnothta einai kanonikopoihmenh ws pros th syxnothta diegershs

---> fourier harmonics vector maker (fourier_vector_maker_harmonics)
Pairnei san input data (mia lista h array apo shmata) kai efarmozei th sunarthsh fourier_signal_normalization_harmonics kai dinei tis armonikes apo
to kanonikopoihmeno shma to input einai ta data kai ta output einai mia lista me to kanonikopoihmeno amplitude twn armonikwn kai mia lista me thn kanonikopoihmenh 
syxnothta twn armonikwn

---> harmonics fourier nrm vector (fourier_nrm_vector_harmonics)
To input einai to path kai efarmozei thn sunarthsh fourier_vector_maker gia olous tous sensors kathe shmatos sto path.
To input einai to path kai to output einai oi concatenated normalized armonikes olwn twn samples sto path

'''

def fourier_harmonics(sample):
    
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
    import numpy as np
    

    ########## pairnei san input sample apo raw shma
    ####### dinei output to amp kai to freq tou kanonikopoihmenou shmatos
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
    ########## pairnei san input data
    ####### dinei output vector kanonikopoihmeno sample me freq
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_normalization_harmonics(sample)[0])
        freq_vector.append(fourier_signal_normalization_harmonics(sample)[1])
    return feature_vector,freq_vector


def fourier_nrm_vector_harmonics(path):

    import numpy as np
    
    ########## pairnei san input path
    ####### dinei output to nrm fourier shma
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
pairnei san input ena shma kai ypologizei to fft tou shmatos
dinei san output to amplitude tou shmatos kai th suxnothta

---> pwelch (pwelch)
pairnei san input ena shma kai ypologizei to pwelch tou shmatos
dinei san output to amplitude tou shmatos kai th suxnothta

---> psd (psd)
pairnei san input ena shma kai ypologizei to psd tou shmatos
dinei san output to amplitude tou shmatos kai th suxnothta

---> spectrogram (spectrogram)
pairnei san input ena shma kai ypologizei to spectrogram tou shmatos
dinei san output to spectrogram tou shmatos

---> wavelet (wavelet)
pairnei san input ena shma kai ypologizei to wavelet tou shmatos
dinei san output to wavelet

---> noise adder (add_noiz)
pairnei san input ena X_set kai prosthetei noise
dinei san output to X_set opou exei prostethei o thorubos
'''

def fourier(sample_sensor):

    import numpy as np
    
    fs = 1/1000
    #the sampling frequency is 1/(seconds in a total experiment time)

    fourier = np.fft.fft(sample_sensor)
    #sample sensor is the value of s2 which is the 
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    #
    power_spectrum = np.log(power_spectrum)
    #
    return power_spectrum,freqs


def pwelch(sample_sensor):

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

    from scipy import signal

    fs = 1000
    # f contains the frequency components
    # S is the PSD
    (f, S) = signal.periodogram(sample_sensor, fs, scaling='density')
    return S,f
    #plt.semilogy(f, S)
    #plt.ylim([1e-14, 1e-3])
    #plt.xlim([0,500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def spectrogram(sample):

    from scipy import signal


    fs = 1000
    f, t, Sxx = signal.spectrogram(sample, fs)
    #plt.pcolormesh(t, f, Sxx, shading='gouraud')
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
    return Sxx

def wavelet(sample):

    import pywt
    import numpy as np

    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
    signal = sample

    # Perform wavelet transform
    wavelet_name = 'db1' # Daubechies wavelet, order 1
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

---> random forest feature elimination with cross validation (rfecv)
pairnei san input to X_train to y_train kai to X_test kai kanei rfecv gia na brei ta kalutera features
to output einai to X_train kai to X_test me ta kalutera features 

---> prinicipal component analysis (pca)
pairnei san input to X_train kai to X_test kai kanei pca gia na krathsei tous grammikous sunduasmous me to megalutero variance
to output einai to X_train me to X_test me ta principal components me to megalutero variance

---> kernel principal component analysis (kpca)
pairnei san input to X_train kai to X_test kai ton kernel kai prwta xrhsimopoiei enan kernel gia na kanei map ta features se ena allo feature space kai meta
kanei pca gia na krathsei ta features me to megalutero variance
to output einai to X_train me to X_test me ta features me tous sunduasmous me to megalutero variance


---> data mixer (data_mixer)
pairnei san input ta X kai Y enos dataset kai ta X kai Y enos deuterou dataset kai to pososto summetoxhs kathe dataset.
Enwnei ta duo datasets basei twn antistoixwn posostwwn summetoxhs kai kanei shuffle ta dedomena to output einai to enwmeno dataset.
'''



def rfecv(X_train,y,X_test):

    
    import pandas as pd

    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeClassifier
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select = 3 )
    rfe.fit(X_train,y)
    feature_list=[]
    for i,col in zip(range(X_train.shape[1]), X_train.columns):
        if rfe.ranking_[i]<2:
            feature_list.append(col)
    X_train_new = pd.DataFrame()
    X_test_new = pd.DataFrame()
    for i in range(0,len(feature_list)):
            X_train_new[f'feature{i}'] = X_train[feature_list[i]]
            X_test_new[f'feature{i}'] = X_test[feature_list[i]]
    X_test = X_test_new
    X_train = X_train_new
    return X_train,X_test

def pca(X_train,X_test):

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
pairnei san input ta y_true,y_pred,model,mode,accuracy kai bgazei to confusion matrix me titlo
to onoma tou montelou kai to accuracy tou. To montelo prepei na einai function kai to mode einai 
eite show eite save.


---> regression results bar chart (regression_results_bar_charts)
the inputs are the model names, the mape values, the standard deviation values, the p-value values as lists and the label on y axis.

---> classification results bar chart (classification_results_bar_charts)
the inputs are the model names, the mape values, the standard deviation values, the p-value values as lists and the label on y axis.

'''


def all_damage_every_sensor_separate(path,index_list):
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
            #ax.set_title(f'{sensor_name} - {label}')

        axs[-1].set_xlabel('Frequency (kHz)')

        fig.text(0.04, 0.5, 'Log Amplitude (V)', va='center', rotation='vertical', fontsize=16)
        fig.suptitle(f'FFT plots of {sensor_name}', fontsize=18)
        fig.tight_layout(rect=[0.06, 0, 1, 0.95])
        plt.show()


def all_damage_every_sensor_together(path,index_list):
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



def parity_plot(y_true,y_pred,model,mode):
    
    
    '''
    prepei to model na einai ws function
    

    '''

    import matplotlib.pyplot as plt

    plt.scatter(y_true,y_pred,color='r')
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


def confusion_matrix_display(y_true,y_pred,model,mode,accuracy):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

   
    if model.__name__ =='svc' : name = 'Support Vector Machines'
    if model.__name__ =='random_forest_clf' : name = 'Random Forest'
    if model.__name__ =='xgb_clf' : name = 'XGB'
    if model.__name__ =='cnn_class' : name = 'CNN'
    cm = confusion_matrix(y_true,y_pred)
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

    # Annotate bars with accuracy ± std and f1 score
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

6)KERNELS




'''




########################################################################

########################################################################

########################################################################

########################################################################

########################################################################


'''

7)EXPERIMENT RUN

---> regression experiment run (regression_model_run)
Pairnei san input to montelo to X_train to y_train to X_test kai y_test kai bgazei san output to 
mae,mape,y_test kai y_pred. To montelo prepei na einai function

---> classification experiment run(classification_model_run)
Pairnei san input to montelo to X_train to y_train to X_test kai y_test kai bgazei san output to 
accuracy,y_test kai y_pred. To montelo prepei na einai function
'''


def regression_model_run(model,X_train,y,X_test,y_true):
    
    from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error

    y_pred = model(X_train,y,X_test)
    #print(y_pred)
    mape = 100*mean_absolute_percentage_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    return mae,mape,y_true,y_pred

def classification_model_run(model,X_train,y,X_test,y_true):

    from sklearn.metrics import accuracy_score
    y_pred = model(X_train,y,X_test)
    #print(y_pred)
    acc = 100*accuracy_score(y_true,y_pred)
    acc = accuracy_score(y_true,y_pred)
    return acc,y_true,y_pred

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################



'''

8)TOOLS FOR TUNING

---> cross validation me leave one out(cross_val_loo)
Pairnei san input to montelo to X kai to y kai kanei leave one out cross validation kai bgazei ta scores kathe fold.
Sto regression to scoring einai  'neg_mean_absolute_percentage_error' kai sto classification einai 'accuracy'


---> grid search me leave one out(grid_search_loo)
Pairnei san input to montelo to X_train kai to y_train kai kanei grid search me leave one out gia na brei tis kaluteres 
parametrous tou montelou. Sto telos kanei print tis kaluteres parametrous.
Analoga to montelo prepei na ruthmistoun oi parametroi pou tha ginei to grid search.
'''

def cross_val_loo(model,X,y):
    '''
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
    ftiaxnw tis parametrous gia to modelo pou tha dwsw sto input
    
    '''
    from sklearn.model_selection import GridSearchCV,LeaveOneOut
    # defining parameter range 
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['rbf','sigmoid','poly']} 

    grid = GridSearchCV(model, param_grid, refit = True, cv=LeaveOneOut(),verbose = False) 

    # fitting the model for grid search 
    grid.fit(X_train, y_train) 

    # print best parameter after tuning 
    print(grid.best_params_)
    print(grid.best_score_)


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################