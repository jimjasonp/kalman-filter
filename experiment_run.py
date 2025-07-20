from helper_functions import y_set, X_set 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from scikeras.wrappers import KerasClassifier, KerasRegressor
from models import (
    linear_regression, svc, random_forest_clf, random_forest_reg,
    keras_mlp_classifier, keras_mlp_regressor,
    keras_cnn_classifier, keras_cnn_regressor,
    keras_lstm_classifier, keras_lstm_regressor
)
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Parameters
n_points_list = [375, 460, 750]
transformation_list = ['none', 'fourier']
noise_levels = [2, 5, 10]

# Paths
balanced = r'C:\Users\jimja\Desktop\thesis\all_datasets\Balanced_Data'
test = r'C:\Users\jimja\Desktop\thesis\all_datasets\test_classification'
random = r'C:\Users\jimja\Desktop\thesis\all_datasets\random_data'



def p_val(y_true, y_pred):
    return pearsonr(y_true, y_pred)[1]


def run_regression_fold(model_fn, X, X_dl, y, train_idx, test_idx, is_dl):
    model = model_fn()
    X_train = X_dl[train_idx] if is_dl else X[train_idx]
    X_test = X_dl[test_idx] if is_dl else X[test_idx]
    model.fit(X_train, y[train_idx])
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y[test_idx], preds)
    pval = p_val(y[test_idx], preds)
    return mape, pval, preds, y[test_idx]


def run_classification_fold(model_fn, X, X_dl, y, train_idx, test_idx, is_dl):
    model = model_fn()
    X_train = X_dl[train_idx] if is_dl else X[train_idx]
    X_test = X_dl[test_idx] if is_dl else X[test_idx]
    model.fit(X_train, y[train_idx])
    preds = model.predict(X_test)
    acc = accuracy_score(y[test_idx], preds)
    f1 = f1_score(y[test_idx], preds, average='macro')
    return acc, f1, preds, y[test_idx]


def regression_experiment_run():
    for n_points in n_points_list:
        all_results = []

        for transformation in transformation_list:
            y_random = y_set(random)['dmg']
            y_data = y_set(balanced)['dmg']

            for noise_percent in noise_levels:
                X_random = X_set(random, transformation, n_points, noise_percent=noise_percent)[0]
                X_data = X_set(balanced, transformation, n_points, noise_percent=noise_percent)[0]

                scaler = StandardScaler()
                X_data = scaler.fit_transform(X_data)
                X_random = scaler.transform(X_random)

                X = np.concatenate((X_data, X_random), axis=0)
                y = np.concatenate((y_data, y_random), axis=0)
                X_dl = np.expand_dims(X, axis=-1)
                input_shape = X.shape[1]

                model_fns = {
                    'LinearRegression': lambda: linear_regression(),
                    'RandomForest': lambda: random_forest_reg(),
                    'MLP': lambda: KerasRegressor(model=keras_mlp_regressor, model__input_shape=(input_shape,), epochs=150, batch_size=64, verbose=0),
                    'CNN': lambda: KerasRegressor(model=keras_cnn_regressor, model__input_shape=(input_shape, 1), epochs=150, batch_size=64, verbose=0),
                    'LSTM': lambda: KerasRegressor(model=keras_lstm_regressor, model__input_shape=(input_shape, 1), epochs=150, batch_size=64, verbose=0),
                }

                cv = KFold(n_splits=5, shuffle=True, random_state=1)

                for name, model_fn in model_fns.items():
                    is_dl = name in ['MLP', 'CNN', 'LSTM']
                    tasks = [
                        delayed(run_regression_fold)(model_fn, X, X_dl, y, train_idx, test_idx, is_dl)
                        for train_idx, test_idx in cv.split(X)
                    ]
                    results_fold = Parallel(n_jobs=2, backend='loky')(tasks)
                    mape_scores, pval_scores, preds_list, y_true_list = zip(*results_fold)

                    all_results.append({
                        'n_points': n_points,
                        'transformation': transformation,
                        'noise_percent': noise_percent,
                        'model': name,
                        'mean_mape': np.mean(mape_scores),
                        'std_mape': np.std(mape_scores),
                        'pval': np.mean(pval_scores),
                        'last_fold_preds': preds_list[-1].tolist(),
                        'last_fold_true': y_true_list[-1].tolist()
                    })

        df = pd.DataFrame(all_results)
        df.to_csv(f'regression_results_n{n_points}.csv', index=False)


def classification_experiment_run():
    for n_points in n_points_list:
        all_results = []

        for transformation in transformation_list:
            y_data = y_set(balanced)['defect']
            y_test = y_set(test)['defect']

            label_map = {label: i for i, label in enumerate(set(y_data) | set(y_test))}
            y_data = np.array([label_map[label] for label in y_data])
            y_test = np.array([label_map[label] for label in y_test])

            for noise_percent in noise_levels:
                X_data = X_set(balanced, transformation, n_points, noise_percent=noise_percent)[0]
                X_test = X_set(test, transformation, n_points, noise_percent=noise_percent)[0]

                scaler = StandardScaler()
                X_data = scaler.fit_transform(X_data)
                X_test = scaler.transform(X_test)

                X = np.concatenate((X_data, X_test), axis=0)
                y = np.concatenate((y_data, y_test), axis=0)
                X_dl = np.expand_dims(X, axis=-1)
                input_shape = X.shape[1]

                model_fns = {
                    'SVC': lambda: svc(),
                    'RandomForest': lambda: random_forest_clf(),
                    'MLP': lambda: KerasClassifier(model=keras_mlp_classifier, model__input_shape=(input_shape,), epochs=150, batch_size=64, verbose=0),
                    'CNN': lambda: KerasClassifier(model=keras_cnn_classifier, model__input_shape=(input_shape, 1), epochs=150, batch_size=64, verbose=0),
                    'LSTM': lambda: KerasClassifier(model=keras_lstm_classifier, model__input_shape=(input_shape, 1), epochs=150, batch_size=64, verbose=0),
                }

                cv = KFold(n_splits=5, shuffle=True, random_state=1)

                for name, model_fn in model_fns.items():
                    is_dl = name in ['MLP', 'CNN', 'LSTM']
                    tasks = [
                        delayed(run_classification_fold)(model_fn, X, X_dl, y, train_idx, test_idx, is_dl)
                        for train_idx, test_idx in cv.split(X)
                    ]
                    results_fold = Parallel(n_jobs=2, backend='loky')(tasks)
                    acc_scores, f1_scores, preds_list, y_true_list = zip(*results_fold)

                    all_results.append({
                        'n_points': n_points,
                        'transformation': transformation,
                        'noise_percent': noise_percent,
                        'model': name,
                        'mean_acc': np.mean(acc_scores),
                        'std_acc': np.std(acc_scores),
                        'f1_macro': np.mean(f1_scores),
                        'last_fold_preds': preds_list[-1].tolist(),
                        'last_fold_true': y_true_list[-1].tolist()
                    })

        df = pd.DataFrame(all_results)
        df.to_csv(f'classification_results_n{n_points}.csv', index=False)


# Run experiments
regression_experiment_run()
classification_experiment_run()
