from helper_functions import X_set,y_set,fourier_nrm_vector,fourier_nrm_vector_harmonics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut,cross_val_score,KFold
from sklearn.neighbors import KNeighborsClassifier


X_clf = fourier_nrm_vector('test_classification')
X_data = fourier_nrm_vector('Balanced_data')

y_clf = y_set('test_classification')['defect']
y_data = y_set('Balanced_data')['defect']

le = LabelEncoder()
y_data = le.fit_transform(y_data)
y_clf = le.transform(y_clf)


scaler = StandardScaler()
X_clf= scaler.fit_transform(X_clf)
X_data = scaler.transform(X_data)

X = np.concatenate((X_data,X_clf),axis=0)
y = np.concatenate((y_data,y_clf),axis=0)


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

xgb = XGBClassifier()
svm=SVC(C=100,gamma=0.001,kernel='rbf')
rf = RandomForestClassifier(n_estimators=500,criterion='entropy')

model_list = [xgb,svm,rf]
for model in model_list:
        for i in range(0,10):
                scores = cross_val_score(model,X,y,cv=KFold(n_splits=10,shuffle=True))
                print(np.mean(scores),np.median(scores))
        print(model)