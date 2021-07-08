import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.feature_extraction import FeatureHasher

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
    stander_scaler = preprocessing.StandardScaler()
    return stander_scaler.fit_transform(X) 

def normalize(X):
    normalizer = preprocessing.Normalizer()
    return normalizer.fit_transform(X) 

def _nans_to_category(X):
    X['barrio'].replace({None: "otro"}, inplace=True)
    X['categoria_de_trabajo'].replace({None: "no_especificada"}, inplace=True)
    X['trabajo'].replace({None: "no_especificado"}, inplace=True)
    return

def _nans_remove(X):
    # eliminar los valores nulos: axis= 0 (filas con nulos)
    X.dropna(axis=0, inplace=True)
    X.reset_index(drop=True, inplace=True)
    return

missings_preprocessing = {
    'category': _nans_to_category,
    'remove': _nans_remove,
}

def init_dataset(X,missings='category'):
    X_aux = X.copy()
    missings_preprocessing[missings](X_aux)
    _transform_types(X_aux)
    X_aux.drop(columns=['educacion_alcanzada'],inplace=True)
    return X_aux

#Transforma los valores de la columna que no pasen un determinado threshold, convirtiendolos en el valor determinado por values
def reduce_by_frequency_occurrence(X,columns_names=['barrio', 'religion','categoria_de_trabajo','estado_marital'],thresholds=[1000,2000,3000,5000],values=['otro','otra','otra','otro']):
    X_aux = X.copy()
    for index, column in enumerate(columns_names):
        values_counts = X[column].value_counts()
        values_to_convert = values_counts[values_counts < thresholds[index]].index
        X_aux[column] = X[column].replace(to_replace = values_to_convert, value=values[index])
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


def select_types(X,types=[]):
    X_aux = X.select_dtypes(include=types)
    return X_aux