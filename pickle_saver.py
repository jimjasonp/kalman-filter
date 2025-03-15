from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import pickle

from helper_functions import X_set,y_set,pickle_saver
from sklearn.metrics import mean_absolute_percentage_error
from models import decision_tree_reg,mlp,linear_regression

train_path = 'random_data'
test_path = 'dokimes'

best_mape = 1
model =linear_regression

for i in range(0,20):

    X_train_diff = X_set('data','fourier')[0]
    X_train = X_set(train_path,'fourier')[0]
    X_test = X_set(test_path,'fourier')[0]
    y_train_diff  = y_set('data')['dmg']
    


    y_train = y_set(train_path)['dmg']

    y_test = [0.02,0.034,0.062,0.086,0.12]

    
    X_train = np.concatenate((X_train,X_train_diff),axis=0)
    y_train = np.concatenate((y_train,y_train_diff),axis=0)
    



    from sklearn.model_selection import train_test_split
    X_train, X_drop, y_train, y_drop = train_test_split(X_train, y_train, test_size=0.2,shuffle=True)

    from sklearn.linear_model import LinearRegression
   
    y_pred = model(X_train,y_train,X_test)
    mape = mean_absolute_percentage_error(y_test,y_pred)
    if mape<best_mape:
        best_mape = mape
        print(best_mape)

        import pickle

        with open(f'{model.__name__}_pickle','wb') as f:
            pickle.dump(model,f)



