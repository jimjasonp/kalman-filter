import matplotlib.pyplot as plt
from helper_functions import X_set,y_set
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from models import svc
from sklearn.preprocessing import StandardScaler
path = 'Balanced_data'




X = X_set(path,'none')[0]
y = y_set(path)['defect']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#X_test = add_noiz(X_test,0.5,0)

y_pred = svc(X_train,y_train,X_test)

acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(acc)