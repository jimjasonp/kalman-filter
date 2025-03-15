from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from helper_functions import x_y_unwanted_remover,rfecv,X_set,y_set,kpca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.gaussian_process.kernels import ExpSineSquared,RBF,Product,Matern,RationalQuadratic
import math

knn = KNeighborsClassifier()
svm = SVC(
    #C = 0.000001 , 
    kernel= 'sigmoid',
    tol = 1e-12,
    #gamma=1,
    coef0=1,
    probability=True,
    shrinking=False,
    cache_size=5000,
    max_iter=-1
    )

transformation = 'fourier'

y = y_set(r'Balanced_data','classification')



sensor2_vector = X_set(r'Balanced_data',transformation)[1]
sensor3_vector = X_set(r'Balanced_data',transformation)[2]
sensor4_vector = X_set(r'Balanced_data',transformation)[3]

X = X_set(r'Balanced_data',transformation)[0]
#X,y = x_y_unwanted_remover(sensor2_vector,sensor3_vector,sensor4_vector,y)
#y = np.array(y['defects'])

X_df = []
X_dm =[]
X_dd =[]
X_ola =[]
X_clean =[]

for i in range(0,len(y)):
    if y[i] == 'dm':X_dm.append(X[i])
    if y[i] == 'df':X_df.append(X[i])
    if y[i] == 'dd':X_dd.append(X[i])
    if y[i] == 'ola':X_ola.append(X[i])
    if y[i] == 'clean':X_clean.append(X[i])





def scatterplot(X_data,y_data):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x,y,z = X_data[0],X_data[1],X_data[2]
    c=[]
    for i in range(0,len(y_data)):
        if y_data[i] == 'dm':c.append(0)
        if y_data[i] == 'df':c.append(1)
        if y_data[i] == 'dd':c.append(2)
        if y_data[i] == 'ola':c.append(3)
        if y_data[i] == 'clean':c.append(4)

    img = ax.scatter(x, y, z,c=c, cmap=plt.hot())
    fig.colorbar(img)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

def scatterplot_height(X_data,y_data):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    c=[]
    for i in range(0,len(y_data)):
        if y_data[i] == 'dm':c.append(0)
        if y_data[i] == 'df':c.append(1)
        if y_data[i] == 'dd':c.append(2)

    x,y,z = X_data[0],X_data[1],c

    img = ax.scatter(x, y, z)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('class')
    plt.show()







def stat_props_calc(X):
    X_mean =[]
    X_std=[]
    X_max =[]
    for number in X:
        X_mean.append(np.mean(number))
        X_std.append(np.std(number))
        X_max.append(np.max(number))

    X_features = pd.DataFrame()

    X_features[0],X_features[1],X_features[2] = X_mean,X_std,X_max
    return X_features

X_df_features = stat_props_calc(X_df)
X_dm_features = stat_props_calc(X_dm)
X_dd_features = stat_props_calc(X_dd)

X_features = stat_props_calc(X)

fig, (ax1, ax2,ax3) = plt.subplots(3)
fig.suptitle('Statistical properties (fft)')

ax1.plot(X_df_features[0])
ax1.plot(X_dm_features[0])
ax1.plot(X_dd_features[0])
ax1.set_title('mean')

ax2.plot(X_df_features[1])
ax2.plot(X_dm_features[1])
ax2.plot(X_dd_features[1])
ax2.set_title('stdev')

ax3.plot(X_df_features[2])
ax3.plot(X_dm_features[2])
ax3.plot(X_dd_features[2])
ax3.set_title('max')

plt.xlabel('sample')
plt.ylabel("metric's value")
plt.show()

'''
paizei kala me custom_kernel =Product(RBF(length_scale=1000),ExpSineSquared(periodicity=1e-4,length_scale=1)) + RBF(length_scale=10) gia n_components=3 kai gia none transformation kai fourier
'''


#custom_kernel =Product(RBF(length_scale=1000),ExpSineSquared(periodicity=1e-4,length_scale=1)) + RBF(length_scale=10)


custom_kernel = Product(RationalQuadratic(alpha=1,length_scale=1e-3),ExpSineSquared(periodicity=1e-4,length_scale=1))



#X = np.concatenate((sensor2_vector,sensor3_vector,sensor4_vector),axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train,X_test = kpca(X_train,X_test,custom_kernel)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

'''dum = DummyClassifier()
dum.fit(X_train,y_train)'''


knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
CM = confusion_matrix(y_test,y_pred)
print(accuracy)
print(CM)

#scatterplot_height(X_train,y_train)
scatterplot(X_train,y_train)



'''



                #scatterplot(X_train,y_train)





                svm = SVC(
                    #C = 0.000001 , 
                    kernel= 'sigmoid',
                    tol = 1e-12,
                    #gamma=1,
                    coef0=1,
                    probability=True,
                    shrinking=False,
                    cache_size=5000,
                    max_iter=-1
                    )




                #del X_train[2]
                #del X_test[2]

X_test = pd.DataFrame(X_test)
X_train = pd.DataFrame(X_train)





y_train_numbers = []

for defect in y_train:
    if defect == 'df':
        y_train_numbers.append(0)
    elif defect == 'dm':
        y_train_numbers.append(1)
    elif defect == 'dd':
        y_train_numbers.append(2)


y_train = y_train_numbers

from sklearn.inspection import DecisionBoundaryDisplay

display = DecisionBoundaryDisplay.from_estimator(svm,X_train,response_method='predict',xlabel='feature_1', ylabel='feature_2',alpha=0.5)


display.ax_.scatter(X_train[0],X_train[1],c=y_train, edgecolor="black")
plt.show()

'''
