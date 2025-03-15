import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np

from helper_functions import y_set,X_set
from models import *
from helper_functions import data_mixer,bar_res_plot,regression_model_run,single_model_result_plot,parity_plot


transformation = 'fourier'

X_data = X_set(r'data',transformation)[0]
X_random_data = X_set(r'random_data',transformation)[0]
X_test = X_set(r'dokimes',transformation)[0]

y_data = y_set(r'data')['dmg']
y_random_data = y_set(r'random_data')['dmg']
y_test = [0.02,0.034,0.062,0.086,0.12]

scaler = StandardScaler()
X_random_data= scaler.fit_transform(X_random_data)
X_data = scaler.transform(X_data)
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)

model_list = [mlp,linear_regression,decision_tree_reg]
name_list = ['mlp','linear regression','decision trees']


max = [] # kai ta duo sets einai full
mid = [] # kai ta duo sets einai misa
min = [] # mono to random_data



#### max krataw olo to random dataset kai olo to original
#### mid krataw to miso random dataset kai to miso original
#### min krataw mono to random
X_train = np.concatenate((X_data,X_random_data),axis=0)
y = np.concatenate((y_data,y_random_data),axis=0)

for model in model_list:
    mae,mape,y_true,y_pred = regression_model_run(model,X_train,y,X_test,y_test)
    max.append(mape) 
    parity_plot(y_true,y_pred,model,'show')
    




X_train,y_train = data_mixer(X_data, y_data,X_random_data, y_random_data,0.5,0.5,'dmg')


for model in model_list:
    mae,mape,y_true,y_pred = regression_model_run(model,X_train,y_train,X_test,y_test)
    mid.append(mape) 
    parity_plot(y_true,y_pred,model,'show')
    





for model in model_list:
    mae,mape,y_true,y_pred = regression_model_run(model,X_random_data,y_random_data,X_test,y_true)
    min.append(mape) 
    parity_plot(y_true,y_pred,model,'show')

bar_res_plot(model_list,min,mid,max,name_list,'regression')


