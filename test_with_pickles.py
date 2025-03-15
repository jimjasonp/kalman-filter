from helper_functions import parity_plot,pca,X_set,y_set

from sklearn.metrics import mean_absolute_percentage_error
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from models import mlp,linear_regression,decision_tree_reg
transformation = 'none'
model = decision_tree_reg
train_path = 'random_data'


min_size = 120

test_path = 'ExtraDataOtherMaterial'
y_test = y_set(test_path)

#test_path = 'dokimes'
#y_test = [0.02,0.034,0.062,0.086,0.12]

#X_train_diff = X_set(r'ExtraDataOtherMaterial','none')[0]
X_train = X_set(train_path,'none')[0]
X_test = X_set(test_path,'none')[0]

X_train = np.concatenate((X_train,X_train_diff),axis=0)

y_train = y_set(train_path)
y_train_diff = y_set(r'ExtraDataOtherMaterial')
y_train = np.concatenate((y_train,y_train_diff),axis=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#X_train,X_test = pca(X_train,X_test)

with open(f'dt_pickle','rb') as f:
    mp = pickle.load(f)



y_pred = mp.predict(X_test)
mape = mean_absolute_percentage_error(y_test,y_pred)
print(mape)
parity_plot(y_test,y_pred,model,'show')


