from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,LeaveOneOut,cross_val_score
from helper_functions import X_set,y_set,fourier_nrm_vector,fourier_nrm_vector_harmonics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression

svm = SVC(C=100,gamma=0.01,kernel='rbf')
lr = LogisticRegression()


path = 'test_classification'



#X = X_set(path,'none')[0]
y = y_set(path)['defect']
X = fourier_nrm_vector_harmonics(path,120)



acc_list  =[]
for i in range(0,100):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    '''X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)'''
    
    svm.fit(X_train,y_train)
    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    acc_list.append(acc)

print(np.mean(acc_list))

