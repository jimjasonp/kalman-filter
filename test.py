from helper_functions import X_set,y_set,fourier_nrm_vector,fourier_nrm_vector_harmonics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut,cross_val_score,KFold
from sklearn.neighbors import KNeighborsClassifier


knn_results = pd.read_csv('knn_results_nrm')
knn_results = knn_results['0']
knn_mean = np.mean(knn_results)
knn_std =  np.std(knn_results)

lr_results = pd.read_csv('lr_class_res')
lr_results = lr_results['0']
lr_mean = np.mean(lr_results)
lr_std =  np.std(lr_results)

svm_results = pd.read_csv('svm_class_res')
svm_results = svm_results['0']
svm_mean = np.mean(svm_results)
svm_std=   np.std(svm_results)


barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

mean_list = [svm_mean,knn_mean,lr_mean]
std_list = [svm_std,knn_std,lr_std]


br1 = np.arange(len(std_list)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

def addlabels(x,y,thesi):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = thesi)

plt.bar(br1, mean_list, color ='r', width = barWidth, 
        edgecolor ='grey', label ='Mean') 
plt.bar(br2, std_list, color ='g', width = barWidth, 
        edgecolor ='grey', label ='Standard Deviation') 


plt.xlabel('Algorithm', fontweight ='bold', fontsize = 15) 
plt.ylabel('Metrics value', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(std_list))], 
        ['SVM', 'KNN', 'LR'])

addlabels(br1,mean_list,'right')
addlabels(br2,std_list,'left')

plt.legend()
plt.show() 

