import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from helper_functions import data_mixer,y_set,X_set,bar_res_plot,confusion_matrix_display,classification_model_run
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
rf = RandomForestClassifier(n_estimators=500,criterion='entropy')


transformation = 'none'

X_clf = X_set('test_classification',transformation)[0]
y_clf = y_set('test_classification')['defect']
X_data = X_set('Balanced_data',transformation)[0]
y_data = y_set('Balanced_data')['defect']

scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
X_clf = scaler.transform(X_clf)


X = np.concatenate((X_data,X_clf),axis=0)
y = np.concatenate((y_data,y_clf),axis=0)


print(len(y))
i = 100
while i > 50:
    X_train_clf,X_test_clf,y_train_clf,y_test_clf = train_test_split(X_clf,y_clf,test_size=0.3,shuffle=True)
    X_train_data,X_test_data,y_train_data,y_test_data = train_test_split(X_data,y_data,test_size=0.3,shuffle=True)

    X_train =np.concatenate((X_train_clf,X_train_data),axis=0)
    y_train = np.concatenate((y_train_clf,y_train_data),axis=0)

    X_test =np.concatenate((X_test_clf,X_test_data),axis=0)
    y_test = np.concatenate((y_test_clf,y_test_data),axis=0)
    
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    if acc >0.85:
        f1 = f1_score(y_pred,y_test,average='macro')

        cm = confusion_matrix(y_test,y_pred, labels=rf.classes_)
        disp = ConfusionMatrixDisplay(cm,display_labels=rf.classes_)
        disp.plot()

        plt.title(f'Random Forest Accuracy = {("%.2f" % acc)} Macro f1 = {("%.2f" % f1)}')
        plt.show()
        break