from helper_functions import new_fourier_nrm_vector,y_set,X_set,grid_search_loo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
balanced_path = 'Balanced_data'
test_path = 'test_classification'

X = X_set(test_path,'none')[0]
y = y_set(test_path)['defect']

print('===========================')
print('=---test classification---=')
for i in range(0,5):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    svm = SVC()
    '''
    svm.fit(X_train,y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    print(acc)'''

    grid_search_loo(svm,X,y)
print('===========================')
print('=---balanced_data---=')

X = X_set(balanced_path,'none')[0]
y = y_set(balanced_path)['defect']
for i in range(0,5):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    svm = SVC()
    '''
    svm.fit(X_train,y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    print(acc)'''

    grid_search_loo(svm,X,y)
print('===========================')
print('=---balanced_and test data---=')

X_balanced = X_set(balanced_path,'none')[0]
y_balanced = y_set(balanced_path)['defect']
X_test = X_set(test_path,'none')[0]
y_test = y_set(test_path)['defect']
X = np.concatenate((X_balanced,X_test),axis=0)
y = np.concatenate((y_balanced,y_test),axis=0)

for i in range(0,5):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    svm = SVC()
    '''
    svm.fit(X_train,y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    print(acc)'''

    grid_search_loo(svm,X,y)
print('===========================')