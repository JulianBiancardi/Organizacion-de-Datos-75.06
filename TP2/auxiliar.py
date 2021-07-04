import pandas as pd
import numpy as np
from collections import Counter
import requests



#--------------------------------------auxiliar functions----------------------------------------#
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

def _transform_misspelled(df):
    df['barrio'] = df['barrio'].str.lower()
    df['barrio'].replace({"cilla riachuelo": "villa riachuelo"}, inplace=True)
    df['categoria_de_trabajo'].replace({"monotibutista": "monotributista"}, inplace=True)
    return

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


#--------------------------------------DATASET FUNCTIONS----------------------------------------#

def get_train_set(missings='category'):
    with requests.get(
        "https://docs.google.com/spreadsheets/d/1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0/export?format=csv"
    ) as r, open("impuestos.csv", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
    df = pd.read_csv("impuestos.csv")
    
    missings_preprocessing[missings](df)
    _transform_misspelled(df)
    _transform_types(df)
    
    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = df.filter(items=["tiene_alto_valor_adquisitivo"]).values.ravel()
    return X,y

def get_holdout_set(missings='category'):
    with requests.get(
        "https://docs.google.com/spreadsheets/d/1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE/export?format=csv"
    ) as r, open("df_holdout.csv", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
    df_holdout = pd.read_csv("df_holdout.csv")
    
    ids = df_holdout.filter(items=["id"]).values.ravel()
    df_holdout.drop(columns=['id','representatividad_poblacional'],inplace=True)
    missings_preprocessing[missings](df_holdout)
    _transform_misspelled(df_holdout)
    _transform_types(df_holdout)
    return ids, df_holdout


def create_prediction(ids, y_pred,file_name=''):
    data = {'ids': ids, 'tiene_alto_valor_adquisitivo': y_pred}
    df = pd.DataFrame(data=data)
    df.to_csv(file_name + '.csv')
    return

#------------------------------------------------------------------------------------------------#

#-----------------------------------PLOTTING METRICS FUNCTIONS----------------------------------#

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


from sklearn import tree
import graphviz
from IPython.display import SVG, display

def plot_tree(model_tree,feature_names):
    dot_data = tree.export_graphviz(model_tree,out_file=None, class_names=['0','1'],filled=True, feature_names=feature_names) 
    graph = graphviz.Source(dot_data, format='png')
    display(SVG(graph.pipe(format='svg')))


#------------------------------------------------------------------------------------------------#