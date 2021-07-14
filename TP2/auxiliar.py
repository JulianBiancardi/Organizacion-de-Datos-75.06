import pandas as pd
import numpy as np
from collections import Counter
import requests

#--------------------------------------auxiliar functions----------------------------------------#
def _transform_misspelled(df):
    df['barrio'] = df['barrio'].str.lower()
    df['barrio'].replace({"cilla riachuelo": "villa riachuelo"}, inplace=True)
    df['categoria_de_trabajo'].replace({"monotibutista": "monotributista"}, inplace=True)
    return

#------------------------------------------------------------------------------------------------#


#--------------------------------------DATASET FUNCTIONS----------------------------------------#

def get_train_set():
    with requests.get(
        "https://docs.google.com/spreadsheets/d/1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0/export?format=csv"
    ) as r, open("impuestos.csv", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
    df = pd.read_csv("impuestos.csv")
    
    _transform_misspelled(df)
    
    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = df.filter(items=["tiene_alto_valor_adquisitivo"]).values.ravel()
    return X,y

def get_holdout_set():
    with requests.get(
        "https://docs.google.com/spreadsheets/d/1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE/export?format=csv"
    ) as r, open("df_holdout.csv", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
    df_holdout = pd.read_csv("df_holdout.csv")
    
    ids = df_holdout.filter(items=["id"]).values.ravel()
    df_holdout.drop(columns=['id','representatividad_poblacional'],inplace=True)
    _transform_misspelled(df_holdout)
    return ids, df_holdout


def create_prediction(ids, y_pred,file_name=''):
    data = {'id': ids, 'tiene_alto_valor_adquisitivo': y_pred}
    df = pd.DataFrame(data=data)
    df.to_csv(file_name + '.csv',index=False)
    return

#------------------------------------------------------------------------------------------------#

#-----------------------------------PLOTTING METRICS FUNCTIONS----------------------------------#

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve,auc,roc_auc_score

def plot_cm(clf_model, X_true, y_true):
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.grid(False)
    plot_confusion_matrix(
        clf_model, X_true, y_true, cmap=plt.cm.Blues, display_labels=['0', '1'], ax=ax
    )
    plt.show()


from sklearn import tree
import graphviz
from IPython.display import SVG, display

def plot_tree(model_tree,feature_names):
    dot_data = tree.export_graphviz(model_tree,out_file=None, class_names=['0','1'],filled=True, feature_names=feature_names) 
    graph = graphviz.Source(dot_data, format='png')
    display(SVG(graph.pipe(format='svg')))


#------------------------------------------------------------------------------------------------#
