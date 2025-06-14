import torch
import torch.nn as nn
import torch.optim as optim
from helper_functions import y_set, X_set
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr
from skorch import NeuralNetClassifier, NeuralNetRegressor
import torch_directml
import sys

# --- Check for GPU via DirectML ---
if torch_directml.device_count() == 0:
    print("No GPU found")
    sys.exit()

# Set device to DirectML for AMD GPU
device = torch_directml.device()

# --- Models ---
class DummyClassifierModel(nn.Module):
    def __init__(self, input_shape):
        super(DummyClassifierModel, self).__init__()
        self.dummy_weight = nn.Parameter(torch.ones(1), requires_grad=False)
    
    def forward(self, x):
        batch_size = x.size(0)
        return self.dummy_weight.repeat(batch_size, 1)

class DummyRegressorModel(nn.Module):
    def __init__(self, input_shape):
        super(DummyRegressorModel, self).__init__()
        self.dummy_weight = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, x):
        batch_size = x.size(0)
        return self.dummy_weight.repeat(batch_size, 1)

class TreeEnsembleClassifier(nn.Module):
    def __init__(self, input_shape):
        super(TreeEnsembleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class TreeEnsembleRegressor(nn.Module):
    def __init__(self, input_shape):
        super(TreeEnsembleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class LinearSVMClassifier(nn.Module):
    def __init__(self, input_shape):
        super(LinearSVMClassifier, self).__init__()
        self.linear = nn.Linear(input_shape, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class LinearRegressionModel(nn.Module):
    def __init__(self, input_shape):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_shape, 1)
    
    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_shape):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class MLPRegressor(nn.Module):
    def __init__(self, input_shape):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# --- Data Loading and Preprocessing ---
transformation = 'none' #'fourier'

X_random = X_set(r"/home/jimja/random_data", transformation)[0]
X_data = X_set(r'/home/jimja/Balanced_Data', transformation)[0]
X_test_cla = X_set(r'/home/jimja/test_classification', transformation)[0]

y_random = y_set(r"/home/jimja/random_data")['dmg']
y_data_reg = y_set(r'/home/jimja/Balanced_Data')['dmg']
y_data_clf = y_set(r'/home/jimja/Balanced_Data')['defect']
y_test_cla = y_set(r'/home/jimja/test_classification')['defect']

scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
X_test_cla = scaler.transform(X_test_cla)
X_random = scaler.transform(X_random)

X_reg = np.concatenate((X_data, X_random), axis=0)
y_reg = np.concatenate((y_data_reg, y_random), axis=0)

X_clf = np.concatenate((X_data, X_test_cla), axis=0)
y_clf = np.concatenate((y_data_clf, y_test_cla), axis=0)

# Choose Task: Regression or Classification
X = X_reg
y = y_reg

# Fix types and shapes
X = X.astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

# Check for NaNs
assert not np.isnan(X).any(), "X contains NaNs"
assert not np.isnan(y).any(), "y contains NaNs"

# --- Custom Scorer ---
def pearson_p(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return pearsonr(y_true, y_pred)[1]

scoring = make_scorer(pearson_p, greater_is_better=False) # regression : 'neg_mean_absolute_percentage_error',make_scorer(p_val) --- classification :'accuracy','f1_macro'

# ---- Classifiers ----
dum_clf = NeuralNetClassifier(
    DummyClassifierModel,
    module__input_shape=X.shape[1],
    max_epochs=10,
    batch_size=64,
    optimizer=optim.Adam,
    device=device,
    verbose=0,
    criterion=nn.BCELoss,
    train_split=None
)
rf_clf = NeuralNetClassifier(
    TreeEnsembleClassifier,
    module__input_shape=X.shape[1],
    max_epochs=100,
    batch_size=64,
    optimizer=optim.Adam,
    device=device,
    verbose=0,
    criterion=nn.BCELoss
)
svm = NeuralNetClassifier(
    LinearSVMClassifier,
    module__input_shape=X.shape[1],
    max_epochs=100,
    batch_size=64,
    optimizer=optim.Adam,
    device=device,
    verbose=0,
    criterion=nn.BCELoss
)
mlp_clf = NeuralNetClassifier(
    MLPClassifier,
    module__input_shape=X.shape[1],
    max_epochs=150,
    batch_size=64,
    optimizer=optim.Adam,
    device=device,
    verbose=0,
    criterion=nn.BCELoss
)

# ---- Regressors ----
dum_reg = NeuralNetRegressor(
    DummyRegressorModel,
    module__input_shape=X.shape[1],
    max_epochs=10,
    batch_size=64,
    optimizer=optim.Adam,
    device=device,
    verbose=0,
    criterion=nn.MSELoss,
    train_split=None
)
rf_reg = NeuralNetRegressor(
    TreeEnsembleRegressor,
    module__input_shape=X.shape[1],
    max_epochs=100,
    batch_size=64,
    optimizer=optim.Adam,
    device=device,
    verbose=0,
    criterion=nn.MSELoss
)
lr = NeuralNetRegressor(
    LinearRegressionModel,
    module__input_shape=X.shape[1],
    max_epochs=100,
    batch_size=64,
    optimizer=optim.Adam,
    device=device,
    verbose=0,
    criterion=nn.MSELoss
)
mlp_reg = NeuralNetRegressor(
    MLPRegressor,
    module__input_shape=X.shape[1],
    max_epochs=150,
    batch_size=64,
    optimizer=optim.Adam,
    device=device,
    verbose=0,
    criterion=nn.MSELoss
)


model = lr  # Choose your model here

# --- Cross Validation ---
cv = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=1)

# --- Output ---
print("Mean Pearson p-value (absolute):", np.mean(np.abs(scores)))
