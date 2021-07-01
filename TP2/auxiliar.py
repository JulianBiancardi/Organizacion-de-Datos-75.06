import pandas as pd
from collections import Counter
import requests

from sklearn.model_selection import train_test_split


def get_train_validation(test_size=0.2, missings='category',feature_engineering=True):
    with requests.get(
        "https://docs.google.com/spreadsheets/d/1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0/export?format=csv"
    ) as r, open("impuestos.csv", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
    df = pd.read_csv("impuestos.csv")
    
    missings_preprocessing[missings](df)
    if (feature_engineering):
        _feature_engineering(df)
    
    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = df.filter(items=["tiene_alto_valor_adquisitivo"])
    return train_test_split(X,y, test_size=test_size, random_state=19, stratify=y)


def get_holdout(missings='category',feature_engineering=True):
    with requests.get(
        "https://docs.google.com/spreadsheets/d/1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE/export?format=csv"
    ) as r, open("df_holdout.csv", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
    df_holdout = pd.read_csv("df_holdout.csv")
    
    ids = df_holdout['id']
    df_holdout.drop(columns=['id','representatividad_poblacional'],inplace=True)
    missings_preprocessing[missings](df_holdout)
    if (feature_engineering):
        _feature_engineering(df_holdout)
    return ids, df_holdout


def create_prediction(_id, y_pred,name=''):
    df = pd.DataFrame(y_pred, columns=['tiene_alto_valor_adquisitivo'])
    print(df)
    df.to_csv(name +'.csv')
    return


def _nans_to_category(df):
    df['barrio'].replace({None: "otro"}, inplace=True)
    df['categoria_de_trabajo'].replace({None: "no_especificada"}, inplace=True)
    df['trabajo'].replace({None: "no_especificado"}, inplace=True)
    return

def _nans_remove(df):
    # eliminar los valores nulos: axis= 0 (filas con nulos)
    # eliminar los valores nulos: axis= 1 (columnas con nulos)
    df.dropna(axis=0, inplace=True)
    return

missings_preprocessing = {
    'category': _nans_to_category,
    'remove': _nans_remove,
}

def _feature_engineering(df):
    df.drop(columns=['educacion_alcanzada'],inplace=True)
    df.loc[df.barrio != "Palermo", "barrio"] = "otro"
    #df.loc[(df.religion != "Cristianismo") & (df.religion != "Judaismo"), "religion"] = "otra" VER ESTO
    _transform_types(df)
    return

def _transform_types(df):
    df['barrio'] = df['barrio'].astype("category")
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].astype("category")
    df['estado_marital'] = df['estado_marital'].astype("category")
    df['genero'] = df['genero'].astype("category")
    df['religion'] = df['religion'].astype("category")
    df['rol_familiar_registrado'] = df['rol_familiar_registrado'].astype("category")
    df['trabajo'] = df['trabajo'].astype("category")
    return


#---------------------------PLOTTING METRICS FUNCTIONS------------------#

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve,auc,roc_auc_score

def plot_cm(clf_model, X_true, y_true):
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.grid(False)
    plot_confusion_matrix(
        clf_model, X_true, y_true, cmap=plt.cm.Blues, display_labels=['0', '1'], ax=ax
    )
    plt.show()

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(15, 10))
    plt.plot(
        fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})'
    )
    plt.scatter(fpr, thresholds)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()