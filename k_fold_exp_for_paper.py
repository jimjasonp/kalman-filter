from helper_functions import y_set,X_set,fourier_nrm_vector,fourier_nrm_vector_harmonics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error,make_scorer
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut,cross_val_score,KFold
from sklearn.dummy import DummyRegressor
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor


lr = LinearRegression()
rf = RandomForestRegressor(random_state=1)
dum = DummyRegressor()
logistic = LogisticRegression()

transformation = 'none'
task ='classification' #classification,regression
model = logistic


X_reg = X_set('random_data',transformation)[0]
X_data = X_set('Balanced_data',transformation)[0]
X_clf = X_set('test_classification',transformation)[0]


y_reg = y_set('random_data')['dmg']
y_data_reg = y_set('Balanced_data')['dmg']
y_data_clf = y_set('Balanced_data')['defect']
y_clf = y_set('test_classification')['defect']


scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
X_clf = scaler.transform(X_clf)
X_reg = scaler.transform(X_reg)

def p_val(x,y):
    return pearsonr(x,y)[1]


if task =='regression':

    X = np.concatenate((X_data,X_reg),axis=0)
    y = np.concatenate((y_data_reg,y_reg),axis=0)
    scoring='neg_mean_absolute_percentage_error'
    custom_score = make_scorer(p_val)

elif task =='classification':
    X = np.concatenate((X_data,X_clf),axis=0)
    y = np.concatenate((y_data_clf,y_clf),axis=0)
    scoring='accuracy'






## ++ mlp regressor


## dokimazo na kalo to modelo san function pou kanei return to model
## dokimazo na kanw manual k fold kai cv giati thelw se kapoio fold na mou dwsei to parity plot


#scores = 'neg_mean_absolute_percentage_error',custom_score

cv = KFold(n_splits=10,shuffle=True,random_state=1)
scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
print(np.mean(np.absolute(scores)))



'''for i in range(0,100):
    cv = KFold(n_splits=10,shuffle=True,random_state=1)
    scores = cross_val_score(rf, X, y, scoring='neg_mean_absolute_percentage_error', cv=cv, n_jobs=-1)
    acc_per_fold = np.absolute(scores)
    mean_acc = np.mean(acc_per_fold)
    std_acc = np.std(acc_per_fold)
    mean_mape_list.append(mean_acc)
    std_mape_list.append(std_acc)

print(mean_mape_list,std_mape_list)'''