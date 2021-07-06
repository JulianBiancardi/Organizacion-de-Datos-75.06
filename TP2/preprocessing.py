import numpy as np
import pandas as pd
from sklearn import preprocessing

#--------------------------------------auxiliar functions----------------------------------------#
def stratify_edad(edad):
    if (edad < 35): 
        return "joven"
    if (edad < 60):
        return "adulto"
    else:
        return "anciano"
    
def stratify_horas_trabajo(horas):
    if (horas < 30): 
        return "menos_30"
    if (horas < 50):
        return "entre_30_50"
    else:
        return "mas_50"
    
def stratify_balance(bolsa):
    if (bolsa > 0):
        return "Positivo"
    if bolsa < 0:
        return "Negativo"
    else:
        return "Neutro"
    
def stratify_educacion_alcanzada(educacion):
    if('preescolar' in educacion):
        return 'preescolar'
    if('grado' in educacion):
        return 'primaria'
    if('universidad' in educacion):
        return 'universidad'
    else:
        return 'secundaria'
#------------------------------------------------------------------------------------------------#   

OHE_encoder = preprocessing.OneHotEncoder(drop='first',sparse=False)

def init_OHE(X):
    columns_to_encode = X.select_dtypes(exclude=['int64']).columns
    OHE_encoder.fit(X[columns_to_encode])
    return

def apply_OHE(X):
    columns_to_encode = X.select_dtypes(exclude=['int64']).columns
    X_aux = X.drop(columns=columns_to_encode)
    X_encoder = OHE_encoder.transform(X[columns_to_encode])
    columns_encoder = OHE_encoder.get_feature_names(columns_to_encode)
    X_aux[columns_encoder] = X_encoder
    return X_aux
    
    
def standarize(X):
    stander_scaler = preprocessing.StandardScaler()
    return stander_scaler.fit_transform(X) 

def normalize(X):
    normalizer = preprocessing.Normalizer()
    return normalizer.fit_transform(X) 


def _nans_to_category(df):
    df['barrio'].replace({None: "otro"}, inplace=True)
    df['categoria_de_trabajo'].replace({None: "no_especificada"}, inplace=True)
    df['trabajo'].replace({None: "no_especificado"}, inplace=True)
    return

def _nans_remove(df):
    # eliminar los valores nulos: axis= 0 (filas con nulos)
    df.dropna(axis=0, inplace=True)
    return

missings_preprocessing = {
    'category': _nans_to_category,
    'remove': _nans_remove,
}

def feature_engineering(X,missings='category'):
    X_aux = X.copy()
    missings_preprocessing[missings](df)
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


def discretizar(X):
    X_aux = X.copy()
    X_aux['balance'] = X_aux["ganancia_perdida_declarada_bolsa_argentina"].apply(stratify_balance)
    X_aux['edad_categorica'] = X_aux["edad"].apply(stratify_edad)
    X_aux['educacion'] = X_aux["educacion_alcanzada"].apply(stratify_educacion_alcanzada)
    X_aux['horas_trabajo'] = X_aux["horas_trabajo_registradas"].apply(stratify_horas_trabajo)
    return X_aux

    