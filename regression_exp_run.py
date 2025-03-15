import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np

from file_opener import y_set,X_set
from models import *
from helper_functions import fourier_std_vector,res_plot,regression_model_run,single_model_result_plot,parity_plot,fourier_std_vector_harmonics


# transformations fourier,none,wavelet,psd,pwelch
transformation = 'fourier'
X_data = X_set(r'data',transformation)[0]
X_random_data = X_set(r'random_data',transformation)[0]
X_dokimes = X_set(r'dokimes',transformation)[0]

#X_data = fourier_std_vector(r'C:\Users\jimja\Desktop\thesis\data')
#X_random_data = fourier_std_vector(r'C:\Users\jimja\Desktop\thesis\random_data')
#X_dokimes = fourier_std_vector(r'C:\Users\jimja\Desktop\thesis\dokimes')

y_data = y_set(r'data')
y_random_data = y_set(r'random_data')

y_true = [0.02,0.034,0.062,0.086,0.12]
#y_true = y_set(r'C:\Users\jimja\Desktop\thesis\Diff_mat')





scaler = StandardScaler()
#scaler = MinMaxScaler()
X_random_data= scaler.fit_transform(X_random_data)
X_data = scaler.transform(X_data)
X_dokimes = scaler.transform(X_dokimes)
X_dokimes = pd.DataFrame(X_dokimes)

model_list = [mlp,linear_regression,decision_tree_reg]
name_list = ['mlp','linear regression','decision trees']


max = [] # kai ta duo sets einai full
mid = [] # kai ta duo sets einai misa
min = [] # mono to random_data

max_preds = []
mid_preds =[]
min_preds =[]



#### max krataw olo to random dataset kai olo to original
X_train = np.concatenate((X_data,X_random_data),axis=0)
y = np.concatenate((y_data,y_random_data),axis=0)
X_test = X_dokimes

for model in model_list:
    mae,mape,y_true,y_pred = regression_model_run(model,X_train,y,X_test,y_true)
    max.append(mape) 
    max_preds.append(y_pred)
    parity_plot(y_true,y_pred,model,'show')
    

#### mid krataw to miso random dataset kai to miso original

X_data_half, X_drop, y_data_half, y_drop = train_test_split(X_data, y_data, test_size=0.5,shuffle=True)
X_random_data_half, X_drop, y_random_data_half, y_drop = train_test_split(X_random_data, y_random_data, test_size=0.5,shuffle=True)
X_train_half = np.concatenate((X_data_half,X_random_data_half),axis=0)
y_half = np.concatenate((y_data_half,y_random_data_half),axis=0)
X_test = X_dokimes

for model in model_list:
    mae,mape,y_true,y_pred = regression_model_run(model,X_train,y,X_test,y_true)
    mid.append(mape) 
    mid_preds.append(y_pred)
    parity_plot(y_true,y_pred,model,'show')
    


#### min krataw mono to random

X_train = X_random_data
y = y_random_data
X_test = X_dokimes

for model in model_list:
    mae,mape,y_true,y_pred = regression_model_run(model,X_train,y,X_test,y_true)
    min.append(mape) 
    min_preds.append(y_pred)
    parity_plot(y_true,y_pred,model,'show')

res_plot(model_list,min,mid,max,name_list)
#single_model_result_plot(mlp,X_train,y,X_test,y_true)


