import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,KBinsDiscretizer,StandardScaler,Normalizer,MinMaxScaler

#--------------------------------------auxiliar functions----------------------------------------#
def _transform_types(df):
    df['educacion_alcanzada'] = df['educacion_alcanzada'].astype("category")
    df['barrio'] = df['barrio'].astype("category")
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].astype("category")
    df['estado_marital'] = df['estado_marital'].astype("category")
    df['genero'] = df['genero'].astype("category")
    df['religion'] = df['religion'].astype("category")
    df['rol_familiar_registrado'] = df['rol_familiar_registrado'].astype("category")
    df['trabajo'] = df['trabajo'].astype("category")
    return
#------------------------------------------------------------------------------------------------#   

OHE_encoder = OneHotEncoder(drop='first',sparse=False)

def init_OHE(X):
    columns_to_encode = X.select_dtypes(exclude=['int64', 'int32']).columns
    OHE_encoder.fit(X[columns_to_encode])
    return

def apply_OHE(X):
    columns_to_encode = X.select_dtypes(exclude=['int64', 'int32']).columns
    X_aux = X.drop(columns=columns_to_encode)
    X_encoder = OHE_encoder.transform(X[columns_to_encode])
    columns_encoder = OHE_encoder.get_feature_names(columns_to_encode)
    X_aux[columns_encoder] = X_encoder
    return X_aux

def apply_ODE(X):
    X_encoder = OrdinalEncoder().fit_transform(X)
    return X_encoder

def standarize(X):
    stander_scaler = StandardScaler()
    return stander_scaler.fit_transform(X) 

def normalize(X):
    normalizer = Normalizer()
    return normalizer.fit_transform(X) 

def scale(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def nans_to_category(X):
    X['barrio'].replace({None: "otro"}, inplace=True)
    X['categoria_de_trabajo'].replace({None: "no_especificada"}, inplace=True)
    X['trabajo'].replace({None: "no_especificado"}, inplace=True)
    return

def init_dataset(X):
    X_aux = X.copy()
    nans_to_category(X_aux)
    _transform_types(X_aux)
    X_aux.drop(columns=['educacion_alcanzada'],inplace=True)
    return X_aux

def reduce_by_frequency(X, columns_names, threshold):
    X_aux = X.copy()
    for index, column in enumerate(columns_names):
        values_counts = X[column].value_counts()
        values_to_convert = values_counts[values_counts < len(X_aux.index)*threshold].index
        X_aux[column] = X[column].replace(to_replace = values_to_convert, value='otros')
    return X_aux

#Discretiza los valores de las columnas mediante kmeans.
def discretize_columns(X,columns_names=['anios_estudiados','edad','horas_trabajo_registradas','ganancia_perdida_declarada_bolsa_argentina'],bins=5,strat='kmeans'):
    X_aux = X.copy()
    encoder = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strat)
    for column in columns_names:
        X_binned = encoder.fit_transform(X[[column]])
        X_aux[column + "_bins"] = X_binned.astype(int)
        X_aux.drop(columns=column,inplace=True)
    return X_aux

def eliminar_features(X, columns_names=[]):
    X_aux = X.copy()
    X_aux.drop(columns=columns_names, inplace=True)
    return X_aux
    