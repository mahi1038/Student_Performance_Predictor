import os

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils import save_object, evaluate_model


class ModelTrainerConfig:
    def __init__(self):
        self.model_trainer_file_path = os.path.join("dataset", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )

        models = {
            "random forest": RandomForestRegressor(),
            "decision tree": DecisionTreeRegressor(),
            "gradient boosting": GradientBoostingRegressor(),
            "linear regressor": LinearRegression(),
            "k-nearest neighbour": KNeighborsRegressor(),
            "xgb regressor": XGBRegressor(),
            "cat boost regressor": CatBoostRegressor(verbose = False),
            "ada boost regressor": AdaBoostRegressor()
        }

        model_report : dict = evaluate_model(X_train, y_train, X_test, y_test, models)

        best_model_score = max(model_report.values())

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        if best_model_score < 0.6:
            print("no model found, do something about it")

       
        best_model.fit(X_train, y_train)
        predicted = best_model.predict(X_test)
        r2_accuracy = r2_score(y_test, predicted)

        save_object(
            self.model_trainer_config.model_trainer_file_path, best_model
        )

        return r2_accuracy




