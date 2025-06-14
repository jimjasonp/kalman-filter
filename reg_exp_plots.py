import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from scipy.stats import pearsonr
from helper_functions import y_set,X_set
from models import random_forest_reg
from helper_functions import data_mixer,bar_res_plot,regression_model_run,parity_plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

transformation = 'none'

X_data = X_set(r'Balanced_data',transformation)[0]
X_random_data = X_set(r'random_data',transformation)[0]

y_data = y_set(r'Balanced_data')['dmg']
y_random_data = y_set(r'random_data')['dmg']

X = np.concatenate((X_data,X_random_data),axis=0)
y = np.concatenate((y_data,y_random_data),axis=0)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=True)

scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mae,mape,y_true,y_pred = regression_model_run(random_forest_reg,X_train,y_train,X_test,y_test)
res= pearsonr(y_test,y_pred)[1]
res = "{:e}".format(res)
plt.scatter(y_true,y_pred,color='r')
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints)
plt.xlabel('Test Values')
plt.ylabel('Predicted Values')
plt.title(f'Random Forest MAPE = {("%.4f" % mape)}% p-value = {res}')
plt.legend(["y_values", "y=x"], loc="lower right")
plt.show()