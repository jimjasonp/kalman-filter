import pickle

def random_forest_reg(X_train,y,X_test):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    rf.fit(X_train,y)
    y_pred = rf.predict(X_test)
    return y_pred

def decision_tree_reg(X_train,y,X_test):
    from sklearn.tree import DecisionTreeRegressor
    dt = DecisionTreeRegressor()
    dt.fit(X_train,y)
    y_pred = dt.predict(X_test)
    return y_pred


def lasso_reg(X_train,y,X_test):
    from sklearn.linear_model import Lasso
    lasso = Lasso(
            alpha=0.000000002,
            max_iter=100000,
            selection='random'
            )
    lasso.fit(X_train,y)
    y_pred = lasso.predict(X_test)
    return y_pred

def svr(X_train,y,X_test):
    from sklearn.svm import SVR
    svr = SVR()
    svr.fit(X_train,y)
    y_pred = svr.predict(X_test)
    return y_pred


def gpr(X_train,y,X_test):
    from sklearn.gaussian_process import GaussianProcessRegressor
    gpr = GaussianProcessRegressor()
    gpr.fit(X_train,y)
    y_pred = gpr.predict(X_test)
    return y_pred



def elastic_net_cv(X_train,y,X_test):
    from sklearn.linear_model import ElasticNetCV
    encv = ElasticNetCV(
                cv= 50,
                eps=1e-6,
                l1_ratio=0.01,  
                max_iter=100000,  
                tol=1e-6,  
                selection='cyclic'
                )
    encv.fit(X_train,y)
    y_pred = encv.predict(X_test)
    return y_pred

def elastic_net(X_train,y,X_test):
    from sklearn.linear_model import ElasticNet
    en = ElasticNet(
                #alpha = 0.000000002,
                l1_ratio=0.01,
                alpha = 0.2,  
                max_iter=100000000,  
                tol=0.000001,  
                selection='random'
                )
    en.fit(X_train,y)
    y_pred = en.predict(X_test)
    return y_pred

def linear_regression(X_train,y,X_test):
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train,y)
    y_pred = lr.predict(X_test)
    #with open('lr_pickle','wb') as f:
    #    pickle.dump(lr,f)
    return y_pred


def ridge_reg(X_train,y,X_test):
    from sklearn.linear_model import Ridge
    rr = Ridge(
        alpha=1,   
        max_iter=1000000, 
        tol=0.0000001, 
        solver='auto',  
        )
    rr.fit(X_train,y)
    y_pred = rr.predict(X_test)
    return y_pred

def knn(X_train,y,X_test):
    from sklearn.neighbors import KNeighborsClassifier
    #from tslearn.metrics import dtw
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y)
    y_pred = knn.predict(X_test)
    return y_pred


def svc(X_train,y,X_test):
    from sklearn.svm import SVC
    svm = SVC(C=100,gamma=0.001,kernel='rbf')
    svm.fit(X_train,y)
    y_pred = svm.predict(X_test)
    return y_pred

def random_forest_clf(X_train,y_train,X_test):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=500,criterion='entropy')
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    return y_pred

def xgb_clf(X_train,y_train,X_test):
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(X_train,y_train)
    y_pred = xgb.predict(X_test)
    return y_pred


def logistic_regression(X_train,y,X_test):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train,y)
    y_pred = lr.predict(X_test)
    return y_pred

def mlp(X_train,y,X_test):
    '''
    paizei kala mono gia scaled !!!!!!!!!!!!!!!!!!!!!
    
    '''
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Flatten,Dense


    mlp = Sequential()
    mlp.add(Dense(256, activation='sigmoid'))
    # Dense layer 2 (128 neurons)
    mlp.add(Dense(128, activation='sigmoid'))
    mlp.add(Dense(64, activation='sigmoid'))
    #mlp.add(Dense(32, activation='sigmoid'))
    # Output layer (10 classes)
    mlp.add(Dense(10, activation='sigmoid'))
    mlp.add(Dense(1, activation='linear'))


    mlp.compile(loss="mean_absolute_error", optimizer="adam")
    history = mlp.fit(X_train, y, epochs=150,verbose=0)
    y_pred = mlp.predict(X_test)
    
    return y_pred.ravel().tolist()




def cnn_reg(X_train,y_train,X_test):

    import tensorflow as tf
    from tensorflow import keras
    from keras import layers  
    from keras.models import Sequential

    model = Sequential([
    #layers.Rescaling(1./255),
    layers.Conv1D(16,3, padding = 'same', activation='relu'),
    layers.MaxPooling1D(),
    layers.Conv1D(32,3,padding='same',activation='relu'),
    layers.MaxPooling1D(),
    layers.Conv1D(64,3,padding='same',activation='relu'),
    layers.MaxPooling1D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss="mean_absolute_error")
    model.fit(X_train,y_train,epochs=100,verbose = 0)
    y_pred = model.predict(X_test)
    
    return y_pred.ravel().tolist()


def cnn_class(X_train,y_train,X_test):

    import tensorflow as tf
    from tensorflow import keras
    from keras import layers  
    from keras.models import Sequential

    model = Sequential([
    #layers.Rescaling(1./255),
    layers.Conv1D(16,3, padding = 'same', activation='relu'),
    layers.MaxPooling1D(),
    layers.Conv1D(32,3,padding='same',activation='relu'),
    layers.MaxPooling1D(),
    layers.Conv1D(64,3,padding='same',activation='relu'),
    layers.MaxPooling1D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(4)
    ])
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),metrics = ['accuracy'])
    model.fit(X_train,y_train,epochs=150,verbose = 0)
    y_pred = model.predict(X_test)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    return y_pred

def mlp_classifier(X_train,y,X_test):

    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Flatten,Dense


    mlp = Sequential()
    mlp.add(Dense(256, activation='sigmoid'))
    mlp.add(Dense(128, activation='sigmoid'))
    mlp.add(Dense(64, activation='sigmoid'))
    mlp.add(Dense(4))

    mlp.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics=['accuracy'])

    mlp.fit(X_train, y, epochs=150,verbose=0)
    y_pred = mlp.predict(X_test)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    return y_pred