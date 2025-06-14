from helper_functions import y_set, X_set
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_absolute_percentage_error
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scikeras.wrappers import KerasClassifier, KerasRegressor
from models import keras_mlp_classifier, keras_mlp_regressor
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, LSTM


# --- Model definitions ---

def keras_cnn_regressor(input_shape):
    model = Sequential([
        Conv1D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(),
        Conv1D(32, 3, padding='same', activation='relu'),
        MaxPooling1D(),
        Conv1D(64, 3, padding='same', activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss="mean_absolute_error")
    return model


def keras_cnn_classifier(input_shape):
    model = Sequential([
        Conv1D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(),
        Conv1D(32, 3, padding='same', activation='relu'),
        MaxPooling1D(),
        Conv1D(64, 3, padding='same', activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dropout(0.2),
        Dense(4)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


def keras_lstm_regressor(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(50, activation="relu"),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


def keras_lstm_classifier(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(50, activation="relu"),
        Dense(4, activation="softmax")
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model


def p_val(y_true, y_pred):
    return pearsonr(y_true, y_pred)[1]


# --- Regression experiment ---
def regression_experiment_run():
    transformation_list = ['none', 'fourier']
    results = []

    for transformation in transformation_list:
        X_random = X_set('random_data', transformation)[0]
        X_data = X_set('Balanced_data', transformation)[0]
        y_random = y_set('random_data')['dmg']
        y_data_reg = y_set('Balanced_data')['dmg']

        scaler = StandardScaler()
        X_data = scaler.fit_transform(X_data)
        X_random = scaler.transform(X_random)

        X = np.concatenate((X_data, X_random), axis=0)
        y = np.concatenate((y_data_reg, y_random), axis=0)
        X_dl = np.expand_dims(X, axis=-1)

        dum_reg = DummyRegressor()
        rf_reg = RandomForestRegressor(n_estimators=500, random_state=1)
        lr = LinearRegression()
        mlp_reg = KerasRegressor(model=keras_mlp_regressor, model__input_shape=(X.shape[1],), epochs=150, batch_size=64, verbose=0)
        cnn_reg = KerasRegressor(model=keras_cnn_regressor, model__input_shape=(X.shape[1], 1), epochs=150, batch_size=64, verbose=0)
        lstm_reg = KerasRegressor(model=keras_lstm_regressor, model__input_shape=(X.shape[1], 1), epochs=150, batch_size=64, verbose=0)
        regressors = [cnn_reg, lstm_reg]

        cv = KFold(n_splits=10, shuffle=True, random_state=1)

        for model in regressors:
            mape_scores = []
            pval_scores = []
            is_dl_model = isinstance(model, KerasRegressor)

            for train_idx, test_idx in cv.split(X):
                X_train = X_dl[train_idx] if is_dl_model else X[train_idx]
                X_test = X_dl[test_idx] if is_dl_model else X[test_idx]

                model.fit(X_train, y[train_idx])
                preds = model.predict(X_test)
                mape = mean_absolute_percentage_error(y[test_idx], preds)
                pval = p_val(y[test_idx], preds)
                mape_scores.append(mape)
                pval_scores.append(pval)

            results.append({
                'transformation': transformation,
                'model': model.__class__.__name__,
                'mean_mape': np.mean(mape_scores),
                'std_mape': np.std(mape_scores),
                'pval': np.mean(pval_scores)
            })

    df = pd.DataFrame(results)
    df.to_csv('results_regression.csv', index=False)
    print("Results saved to 'results_regression.csv'")


# --- Classification experiment ---
def classification_experiment_run():
    transformation_list = ['none', 'fourier']
    results = []

    for transformation in transformation_list:
        X_data = X_set('Balanced_data', transformation)[0]
        X_test_cla = X_set('test_classification', transformation)[0]
        y_data_clf = y_set('Balanced_data')['defect']
        y_test_cla = y_set('test_classification')['defect']

        label_map = {label: i for i, label in enumerate(set(y_data_clf) | set(y_test_cla))}
        y_data_clf = np.array([label_map[label] for label in y_data_clf])
        y_test_cla = np.array([label_map[label] for label in y_test_cla])

        scaler = StandardScaler()
        X_data = scaler.fit_transform(X_data)
        X_test_cla = scaler.transform(X_test_cla)

        X = np.concatenate((X_data, X_test_cla), axis=0)
        y = np.concatenate((y_data_clf, y_test_cla), axis=0)
        X_dl = np.expand_dims(X, axis=-1)

        dum_clf = DummyClassifier()
        rf_clf = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=1)
        svm = SVC(C=100, gamma=0.001, kernel='rbf', random_state=1)
        mlp_clf = KerasClassifier(model=keras_mlp_classifier, model__input_shape=(X.shape[1],), epochs=150, batch_size=64, verbose=0)
        cnn_clf = KerasClassifier(model=keras_cnn_classifier, model__input_shape=(X.shape[1], 1), epochs=150, batch_size=64, verbose=0)
        lstm_clf = KerasClassifier(model=keras_lstm_classifier, model__input_shape=(X.shape[1], 1), epochs=150, batch_size=64, verbose=0)
        classifiers = [ cnn_clf, lstm_clf]

        cv = KFold(n_splits=10, shuffle=True, random_state=1)

        for model in classifiers:
            acc_scores = []
            f1_scores = []
            is_dl_model = isinstance(model, KerasClassifier)

            for train_idx, test_idx in cv.split(X):
                X_train = X_dl[train_idx] if is_dl_model else X[train_idx]
                X_test = X_dl[test_idx] if is_dl_model else X[test_idx]

                model.fit(X_train, y[train_idx])
                preds = model.predict(X_test)
                acc = accuracy_score(y[test_idx], preds)
                f1 = f1_score(y[test_idx], preds, average='macro')
                acc_scores.append(acc)
                f1_scores.append(f1)

            results.append({
                'transformation': transformation,
                'model': model.__class__.__name__,
                'mean_acc': np.mean(acc_scores),
                'std_acc': np.std(acc_scores),
                'f1_macro': np.mean(f1_scores)
            })

    df = pd.DataFrame(results)
    df.to_csv('results_classification.csv', index=False)
    print("Results saved to 'results_classification.csv'")


# Run both experiments
regression_experiment_run()
classification_experiment_run()
