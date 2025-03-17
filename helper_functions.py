
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
            defect_list.append('ola')
        elif df_defect !=0 and dm_defect ==0 and dd_defect ==0:
            defect_list.append('df')
        elif df_defect ==0 and dm_defect !=0 and dd_defect ==0:
            defect_list.append('dm')
        elif df_defect ==0 and dm_defect ==0 and dd_defect !=0:
            defect_list.append('dd')
        else:
            defect_list.append('ola')
        
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

Ta duo parakatw functions leitourgoun mazi gia na parw ta harmonics kathe shmatos
gia na parw tis armonikes xrhsimopoiw to fourier_std_vector_harmonics kai bazw san input to path kai to minimum size pou thelw na exw
sthn perioxh pou tha epileksw na krathsw kai mou dinei tis times amplitude kai suxnothtas gia auth thn perioxh

---> fourier signal normalization harmonics (fourier_signal_standardization_harmonics)
To input einai ena raw shma to opoio to kanonikopoiw kai meta thetw kapoia oria pou antistoixizontai se mia perioxh apo ligo prin ews ligo meta thn prwth kai  
ligo prin ews ligo meta thn deuterh armonikh tou kanonikopoihmenou shmatos. Epishs efarmozw ena savgol filter gia na kanw pio smooth to shma
To output einai ena array me tis times amplitude kai ena array me tis syxnothtes twn duo armonikwn

---> fourier normalized signal with harmonics(fourier_std_vector_harmonics)
To input einai to path kai to minimum size pou krataw dhladh h elaxisth posothta timwn pou krataw gia tis duo armonikes.
Prwta pairnw to shma kai to kanonikopoiw kai meta trexw to fourier_signal_standardization_harmonics gia na parw tis armonikes tou. 
Epeidh kathe shma diaferei ligo, oi times max syxnothtas kai max amplitude diaferoyn ligo opote einai fysiologiko oti den pairnw ton idio arithmo datapoints se kathe armonikh opote 
orizw ena katwtato orio to opoio an to yperbainei kapoia periptwsh tote afairw tyxaies times apo to zeygos armonikwn wste na exw osa datapoints osa to orio.
To output einai ena array me tis times tou amplitude kai tou frequency gia tis duo prwtes armonikes tou kanonikopoihmenou shamtos

'''



def fourier_signal_normalization_harmonics(sample):

    import numpy as np

    
    ########## pairnei san input sample apo raw shma
    ####### dinei output to amp kai to freq tou kanonikopoihmenou shmatos
    amp= fourier(sample)[0]
    freq= fourier(sample)[1]

    amp_list =[]
    freq_list =[]
    bound = int(0.5*len(amp))
    max_amp = -max(amp)
    max_freq = abs(freq[amp.argmax()])

    #
    for i in range(0,bound):
        if freq[i]/max_freq>0.8 and freq[i]/max_freq<1.2 or freq[i]/max_freq>1.8 and freq[i]/max_freq<2.2:
            amp_list.append(amp[i]/max_amp)
            freq_list.append(freq[i]/max_freq)
    
    freq = np.array(freq_list)
    amp = np.array(amp_list)
    from scipy.signal import savgol_filter
    
    amp = savgol_filter(amp,5,3)

    return amp,freq



def fourier_nrm_vector_harmonics(path,min_size):

    import numpy as np
    

    ########## pairnei san input path
    ####### dinei output vector kanonikopoihmeno sample me freq
    from helper_functions import X_set 
    data= X_set(path,'none')[0]
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_normalization_harmonics(sample)[0])
        freq_vector.append(fourier_signal_normalization_harmonics(sample)[1])
    #### epeidh exw balei thn if sth sunarthsh fourier kathe sample mesa sto feature vector den exei idio megethos
    ### prepei kathe sample na exei idio megethos gia auto stis periptwseis pou exw parapanw times afairw kapoies
    
    min_size_feature_vector =[]
    min_size_freq_vector =[]

    for sample in feature_vector:
        sample = np.random.choice(sample, size=min_size, replace=False)
        min_size_feature_vector.append(sample)

    for sample in freq_vector:
        sample = np.random.choice(sample, size=min_size, replace=False)
        min_size_freq_vector.append(sample)
    
    feature_vector = min_size_feature_vector
    freq_vector = min_size_freq_vector

    vector = np.concatenate((feature_vector,freq_vector),axis=1)
    
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

---> bar plots specific for regression or classification, pairnei data sizes (bar_res_plot)
Gia to mode regression pairnei san input ta modela pou etreksa kai tis times twn mape twn diaforwn megethwn tou dataset pou etreksan (min,mid,max) kai ta onomata twn montelwn pou etreksan
kai dinei san output ta bar plots twn mape gia kathe montelo. Gia to mode classification kanei to idio alla anti gia mape dinei times accuracy

---> parity plots it can either save or show the plot (parity_plot)
pairnei san input to y_test to y_pred to montelo kai to mode dhladh an thelw na kanw save h aplws na dw to plot
bgazei to parity plot tou y_test me to y_pred kai eite to kanei save eite to deixnei


---> 3d scatter plot for classification (scatterplot_3d_classification)
pairnei san input to x kai to y kai bgazei ena 3d scatter plot twn triwn prwtn features tou x me tis antistoixes times y
oi times tou y einai to xrwma tou kathe shmeiou

---> 2d plot twn shmatwn kathe sensora (separate_sensors_plot)
pairnei to onoma tou defect kai to shma kathe sensora ksewxwrista kai to plotarei
oi times y einai to amplitude kai to x einai to sample

---> 2d plot tou sunolikou shmatos (total_normalized_scatter)
pairnei san input to onoma tou defect kai ena sample apo concatenated normalized 
shma kai dinei san output to plot opou o aksonas y einai to normalized amplitude 
kai o x einai to datapoint 

---> 2d scatter tou sunolikou shmatos(total_normalized_plot)
pairnei san input to onoma tou defect kai ena sample apo
concatenated normalized shma kai dinei san output to scatter opou o aksonas 
y einai to normalized amplitude kai o x einai h suxnothta 


---> confusion matrix gia to classification task (confusion_matrix_display)

pairnei san input ta y_true,y_pred,model,mode,accuracy kai bgazei to confusion matrix me titlo
to onoma tou montelou kai to accuracy tou. To montelo prepei na einai function kai to mode einai 
eite show eite save.

---> 2d plot gia decision boundaries (decision_bounds_plot)

Pairnei san input to X_train to y_train kai to montelo pou etreksa kai bgazei ena 2d plot twn decision boundaries.
Prepei prwta na exw kanei fit to montelo kai oi classes na einai arithmoi
'''

def bar_res_plot(model_list,min,mid,max,name_list,mode):
    
    
    import numpy as np
    import matplotlib.pyplot as plt

    X_axis = np.arange(len(model_list)) 

    plt.bar(X_axis - 0.25, min, 0.2, label = '75 training samples')
    
    #for index, value in enumerate(min):
    #    plt.text(value, index,str(value))
    
    plt.bar(X_axis , mid, 0.2, label = '112 training samples')
    #for index, value in enumerate(mid):
    #    plt.text(value, index,str(value))
    
    plt.bar(X_axis + 0.25 , max, 0.2, label = '225 training samples')
    #for index, value in enumerate(max):
    #    plt.text(value, index,str(value))
    
    plt.xticks(X_axis, name_list)
    plt.xlabel("Models")
    if mode =='regression':
        plt.ylabel("Mean absolute Percentage error")
        plt.title(f"MAPE of models with different training sizes ")
    if mode =='classification':
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy of models with different training sizes ")
    plt.legend() 
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
    plt.title(f'Parity plot of {name}')
    plt.legend(["y_values", "y=x"], loc="lower right")
    if mode=='save':
        plt.savefig(f'{name}_parity_plot.png')
        plt.close('all')
        plt.clf()
    elif mode =='show':
        plt.show()

def scatterplot_3d_classification(X_data,y_data):
    

    '''
    paizei an exw kanei prwta pca 
    
    
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    y_data = np.array(y_data)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(projection='3d')
    x,y,z = X_data[0],X_data[1],X_data[2]
    c=[]
    for i in range(0,len(y_data)):
        if y_data[i] == 'dm':c.append(0)
        if y_data[i] == 'df':c.append(1)
        if y_data[i] == 'dd':c.append(2)
        if y_data[i] == 'ola':c.append(3)
        if y_data[i] == 'clean':c.append(4)

    img = ax.scatter(x, y, z,c=c, cmap=plt.hot())
    fig.colorbar(img)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def separate_sensors_plot(defect,sample_s2,sample_s3,sample_s4):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(sample_s2)
    plt.plot(sample_s3)
    plt.plot(sample_s4)
    plt.title(f'{defect}')
    plt.legend(['s2','s3','s4'],loc = 'lower right')
    plt.ylabel('normalized amplitude')
    plt.xlabel('sample')
    '''plt.savefig(f'{defect}_all_sensors_plot.png')
    plt.close('all')
    plt.clf()'''
    plt.show()



def total_normalized_plot(defect,sample_total):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(sample_total[0])
    plt.title(f'{defect}')
    plt.ylabel('normalized amplitude')
    plt.xlabel('sample')
    plt.show()



def total_normalized_scatter(defect,sample_total):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(sample_total[1],sample_total[0])
    plt.title(f'{defect}')
    plt.ylabel('normalized amplitude')
    plt.xlabel('sample')
    plt.show()

def confusion_matrix_display(y_true,y_pred,model,mode,accuracy):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

   
    if model.__name__ =='svc' : name = 'Support Vector Machines'
    if model.__name__ =='knn' : name = 'K Nearest Neighbors'
    if model.__name__ =='logistic_regression' : name = 'Logistic Regression'

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


def decision_bounds_plot(X_train,y_train,model):

    '''
    thelei oi classes na einai arithmoi
    '''

    import matplotlib.pyplot as plt
    from sklearn.inspection import DecisionBoundaryDisplay

    display = DecisionBoundaryDisplay.from_estimator(model,X_train,response_method='predict',xlabel='feature_1', ylabel='feature_2',alpha=0.5)
    display.ax_.scatter(X_train[0],X_train[1],c=y_train, edgecolor="black")
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
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_percentage_error', cv=cv, n_jobs=-1)
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


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################



''''


check gia na ta prosthesw

'''



########################################################
########################################################
########################################################
########################################################

''''


GIA PETAMA TA BLEPW PRIN TA SBHSW

'''
def single_model_result_plot(model,X_train,y,X_test,y_true):

    import matplotlib.pyplot as plt
    plt.plot(regression_model_run(model,X_train,y,X_test,y_true)[2],marker = 'o')
    plt.plot(regression_model_run(model,X_train,y,X_test,y_true)[3],linestyle='dashed',marker = 'o')
    plt.xlabel("sample")
    plt.ylabel("y value")
    plt.title(f" Predicted and true value of samples using Linear Regression")
    plt.legend(["y_test", "y_pred"], loc="lower right")
    plt.show()





def x_y_unwanted_remover(sensor2,sensor3,sensor4,y):

    import numpy as np
    
    index_remove_list =[]
    for i in range(0,len(y)):
        if y[i] =='clean' or y[i] =='ola':
            index_remove_list.append(i)
    index_remove_list.reverse()

    for i in index_remove_list:
        del sensor2[i]
        del sensor3[i]
        del sensor4[i]
        y =  y.drop([i])
    
    y = np.array(y)
    return sensor2,sensor3,sensor4,y