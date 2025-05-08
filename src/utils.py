import os
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

def  save_object(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok = True)

    with open(file_path, 'wb') as file_obj:
        dill.dump(obj, file_obj)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    for i in range(len(list(models))):
        model = list(models.values())[i]

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        report = {}

        report[list(models.keys())[i]] = test_model_score

    return report

def load_object(file_path):
    with open(file_path, "rb") as file:
        return dill.load(file)
