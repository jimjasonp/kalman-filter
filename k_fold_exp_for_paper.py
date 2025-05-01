from helper_functions import y_set,X_set,fourier_nrm_vector,fourier_nrm_vector_harmonics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error,make_scorer
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut,cross_val_score,KFold
from scipy.stats import pearsonr
from models import keras_mlp_regressor,keras_mlp_classifier
from sklearn.dummy import DummyRegressor,DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scikeras.wrappers import KerasClassifier,KerasRegressor


transformation = 'none'
task ='regression' #classification,regression

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

# ---- Classifiers ----
dum_clf = DummyClassifier()
rf_clf = RandomForestClassifier(n_estimators=500,criterion='entropy')
svm =SVC(C=100,gamma=0.001,kernel='rbf')
mlp_clf = KerasClassifier(model=keras_mlp_classifier,model__input_shape=(X.shape[1],),epochs=150,batch_size=64,verbose=0)

# ---- Regressors ----
dum_reg = DummyRegressor()
rf_reg = RandomForestRegressor(n_estimators=500,criterion='entropy')
lr = LinearRegression()
mlp_reg = KerasRegressor(model=keras_mlp_regressor,model__input_shape=(X.shape[1],),epochs=150,batch_size=64,verbose=0)

model = mlp_reg

## dokimazo na kalo to modelo san function pou kanei return to model
## dokimazo na kanw manual k fold kai cv giati thelw se kapoio fold na mou dwsei to parity plot
## thelw na mou bgazei san metric kai accuracy kai f1-macro kai antoistoixa mape kai p value

#scores = 'neg_mean_absolute_percentage_error',custom_score

cv = KFold(n_splits=10,shuffle=True,random_state=1)
scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
print(np.mean(np.absolute(scores)))
