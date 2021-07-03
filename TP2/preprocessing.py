import numpy as np
import pandas as pd
from sklearn import preprocessing

#--------------------------------------auxiliar functions----------------------------------------#
def stratify_edad(edad):
    if (edad < 20):
        return "joven"
    if (edad < 60):
        return "adulto"
    else:
        return "anciano"
    
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

def apply_OHE(X):
    return pd.get_dummies(X, drop_first=True, columns=None)

def standarize(X):
    stander_scaler = preprocessing.StandardScaler()
    return stander_scaler.fit_transform(X) 

def normalize(X):
    normalizer = preprocessing.Normalizer()
    return normalizer.fit_transform(X) 


def reduce_by_frequency_occurrence(X,columns_names=[],thresholds=[],values=[]):
    #Transforma los valores de la columna que no pasen un determinado threshold, convirtiendolos en el valor determinado por values
    #Ej: X_reduced = pr.reduce_by_frequency_occurrence(X,columns_names=['barrio','religion'],thresholds=[1000,400],values=['otro','otra'])
    X_aux = X.copy()
    for index, column in enumerate(columns_names):
        values_counts = X[column].value_counts()
        values_to_convert = values_counts[values_counts < thresholds[index]].index
        X_aux[column] = X[column].replace(to_replace = values_to_convert, value=values[index])
    return X_aux
        
    
def feature_engineering(X):
    X_aux = X.copy()
    X_aux ['balance'] = X_aux ["ganancia_perdida_declarada_bolsa_argentina"].apply(stratify_balance)
    X_aux = reduce_by_frequency_occurrence(X_aux,columns_names=['barrio'],thresholds=[1000],values=['otro']) #hay un problema con religion en holdout, creo que esto esta mal hacerlo 
    X_aux.drop(columns=['educacion_alcanzada'],inplace=True)
    
    return X_aux 

