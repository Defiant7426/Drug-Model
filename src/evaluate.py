# Código de Evaluación - Modelo de Riesgo de Default en un Banco de Corea
############################################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os


# Cargar la tabla transformada
def eval_model(filename):
    y_test = pd.read_csv(os.path.join('../data/processed', filename)).squeeze()
    print('y_test cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    y_test_pred = pd.read_csv("../data/scores/final_score.csv").squeeze()
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_test_pred)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_test_pred)
    print("Accuracy: ", accuracy_test)


# Validación desde el inicio
def main():
    df = eval_model('drug_y_test.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()