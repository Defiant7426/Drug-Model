# Código de Entrenamiento - Modelo de Riesgo de Default en un Banco de Corea
############################################################################


import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import os


# Cargar la tabla transformada
def read_file_csv(x_train, y_train):
    x_train = pd.read_csv(os.path.join('../data/processed', x_train)) 
    print('cargado x_train')
    y_train = pd.read_csv(os.path.join('../data/processed', y_train)).squeeze()
    print('cargado y_train')
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(x_train,y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(rfc, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('drug_x_train.csv', 'drug_y_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()