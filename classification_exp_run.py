import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from helper_functions import data_mixer,y_set,X_set,bar_res_plot,confusion_matrix_display,classification_model_run
from models import *
from sklearn.metrics import f1_score
transformation = 'fourier'

X_clf = X_set('test_classification',transformation)[0]
y_clf = y_set('test_classification')['defect']
X_data = X_set('Balanced_data',transformation)[0]
y_data = y_set('Balanced_data')['defect']

scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
X_clf = scaler.transform(X_clf)

le = LabelEncoder()
y_clf = le.fit_transform(y_clf)
y_data = le.fit_transform(y_data)

X = np.concatenate((X_data,X_clf),axis=0)
y = np.concatenate((y_data,y_clf),axis=0)


X_train_clf,X_test_clf,y_train_clf,y_test_clf = train_test_split(X_clf,y_clf,test_size=0.3,shuffle=True)
X_train_data,X_test_data,y_train_data,y_test_data = train_test_split(X_data,y_data,test_size=0.3,shuffle=True)

X_train =np.concatenate((X_train_clf,X_train_data),axis=0)
y_train = np.concatenate((y_train_clf,y_train_data),axis=0)

X_test =np.concatenate((X_test_clf,X_test_data),axis=0)
y_test = np.concatenate((y_test_clf,y_test_data),axis=0)

'''scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_clf = scaler.transform(X_train_clf)
X_test_clf = scaler.transform(X_test_clf)'''

model_list = [svc,random_forest_clf,xgb_clf]
name_list = ['svc','random_forest_clf','xgb_clf']

max = [] # kai ta duo sets einai full
mid = [] # kai ta duo sets einai misa
min = [] # mono to random_data

#### max krataw olo to random dataset kai olo to original
#### mid krataw to miso random dataset kai to miso original
#### min krataw mono to random


for model in model_list:
    acc,y_true,y_pred = classification_model_run(model,X_train,y_train,X_test,y_test)
    max.append(acc)
    f1 = f1_score(y_pred,y_test,average='macro')
    print(f1) 
    confusion_matrix_display(y_true,y_pred,model,'show',acc)

X_train_half,X_drop,y_train_half,y_drop = train_test_split(X_train,y_train,test_size=0.5,shuffle=True)

for model in model_list:
    acc,y_true,y_pred = classification_model_run(model,X_train_half,y_train_half,X_test,y_test)
    mid.append(acc) 
    f1 = f1_score(y_pred,y_test,average='macro')
    print(f1)
    confusion_matrix_display(y_true,y_pred,model,'show',acc)


for model in model_list:
    acc,y_true,y_pred = classification_model_run(model,X_train_clf,y_train_clf,X_test,y_test)
    min.append(acc)
    f1 = f1_score(y_pred,y_test,average='macro')
    print(f1)
    confusion_matrix_display(y_true,y_pred,model,'show',acc)

bar_res_plot(model_list,min,mid,max,name_list,'classification')