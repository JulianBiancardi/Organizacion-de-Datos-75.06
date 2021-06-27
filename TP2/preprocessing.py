import pandas as pd
from collections import Counter
import requests

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def get_train_validation(test_size=0.2):
    with requests.get(
        "https://docs.google.com/spreadsheets/d/1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0/export?format=csv"
    ) as r, open("impuestos.csv", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
    df = pd.read_csv("impuestos.csv")
    transformar_nans(df)
    transformar_tipos(df)
    
    X = df.drop("tiene_alto_valor_adquisitivo", axis=1)
    y = df.filter(items=["tiene_alto_valor_adquisitivo"])

    return train_test_split(X,y, test_size=test_size, random_state=19, stratify=y)

def transformar_nans(df):
    df['categoria_de_trabajo'].replace({None: "no_especificada"}, inplace=True)
    df['trabajo'].replace({None: "no_especificado"}, inplace=True)
    return

def transformar_tipos(df):
    df['barrio'] = df['barrio'].astype("category")
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].astype("category")
    df['educacion_alcanzada'] = df['educacion_alcanzada'].astype("category")
    df['estado_marital'] = df['estado_marital'].astype("category")
    df['genero'] = df['genero'].astype("category")
    df['religion'] = df['religion'].astype("category")
    df['rol_familiar_registrado'] = df['rol_familiar_registrado'].astype("category")
    df['trabajo'] = df['trabajo'].astype("category")
    return


def apply_OHE(X):
    X = pd.get_dummies(X, drop_first=True, columns=None)

    return X

def estandarize(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)  
    return X

def normalize(X):
    normalizer = preprocessing.Normalizer()
    X = normalizer.fit_transform(X) 
    return X