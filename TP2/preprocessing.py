import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

def apply_OHE(X):
    X = pd.get_dummies(X, drop_first=True, columns=None)
    return X

def standarize(X):
    stander_scaler = preprocessing.StandardScaler()
    X = stander_scaler.fit_transform(X)  
    return X

def range_scale(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)  
    return X

def robust_scale(X):
    robust_scaler = preprocessing.RobustScaler()
    X = robust_scaler.fit_transform(X)  
    return X

def normalize(X):
    normalizer = preprocessing.Normalizer()
    X = normalizer.fit_transform(X) 
    return X

def apply_PCA(X,n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return X