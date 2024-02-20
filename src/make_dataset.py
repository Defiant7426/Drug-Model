# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename)) 
    print(filename, ' cargado correctamente')
    return df

def label_encoder(datos_categoria, df):
    le = LabelEncoder()
    df[datos_categoria]=le.fit_transform(df[datos_categoria])


# Realizamos la transformación de datos
def data_preparation(df):
    variables = ["Sex","BP","Cholesterol","Na_to_K","Drug"]
    for i in variables:
        label_encoder(i, df)
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, filename):
    df.to_csv(os.path.join('../data/processed/', filename), index = False)
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    df = read_file_csv("drug200.xls")
    df_preparado = data_preparation(df)
    x = df_preparado.drop("Drug",axis=1)
    y = df_preparado.Drug
    x_train,x_test,y_train,y_test = train_test_split(x ,y ,test_size= 0.2,random_state = 42,shuffle = True)
    data_exporting(x_train, "drug_x_train.csv")
    data_exporting(x_test, "drug_x_test.csv")
    data_exporting(y_train, "drug_y_train.csv")
    data_exporting(y_test, "drug_y_test.csv")
    
if __name__ == "__main__":
    main()