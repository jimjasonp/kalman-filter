import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from helper_functions import data_mixer,y_set,X_set,bar_res_plot,confusion_matrix_display,classification_model_run,fourier_nrm_vector
from models import *

'''transformation = 'fourier'
X = X_set('test_classification',transformation)[0]
X_data = X_set('Balanced_data',transformation)[0]'''

X = fourier_nrm_vector('test_classification')
X_data = fourier_nrm_vector('Balanced_data')

y = y_set('test_classification')['defect']
y_data = y_set('Balanced_data')['defect']

X_clf,X_test,y_clf,y_test = train_test_split(X,y,test_size=0.3,shuffle=True)

le = LabelEncoder()
y_data = le.fit_transform(y_data)
y_test = le.transform(y_test)
y_clf = le.transform(y_clf)


scaler = StandardScaler()
X_clf= scaler.fit_transform(X_clf)
X_data = scaler.transform(X_data)
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)

model_list = [svc,xgb_clf,random_forest_clf]
name_list = ['svc','xgb','ramdom_forest']

max = [] # kai ta duo sets einai full
mid = [] # kai ta duo sets einai misa
min = [] # mono to random_data

#### max krataw olo to random dataset kai olo to original
#### mid krataw to miso random dataset kai to miso original
#### min krataw mono to random

X_train = np.concatenate((X_data,X_clf),axis=0)
y = np.concatenate((y_data,y_clf),axis=0)


for model in model_list:
    acc,y_true,y_pred = classification_model_run(model,X_train,y,X_test,y_test)
    max.append(acc) 
    confusion_matrix_display(y_true,y_pred,model,'show',acc)

X_train,y_train = data_mixer(X_data, y_data,X_clf, y_clf,0.3,0.8)


for model in model_list:
    acc,y_true,y_pred = classification_model_run(model,X_train,y_train,X_test,y_test)
    mid.append(acc) 
    confusion_matrix_display(y_true,y_pred,model,'show',acc)


for model in model_list:
    acc,y_true,y_pred = classification_model_run(model,X_clf,y_clf,X_test,y_test)
    min.append(acc) 
    confusion_matrix_display(y_true,y_pred,model,'show',acc)

bar_res_plot(model_list,min,mid,max,name_list,'classification')