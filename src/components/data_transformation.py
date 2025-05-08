import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object


class DataTransformationConfig:
    
    def __init__(self):
        self.preprocessor_obj_file_path: str = os.path.join('dataset', 'preprocessor.pkl')
        

class DataTransformation:

    '''
    this function is responsible for data transformation
    '''

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    



    def get_transformer_obj(self):
        numerical_features = ['writing score', 'reading score']
        categorical_features = [
            'gender',
            'race ethnicity',
            'parental level of education',
            'lunch',
            'test preparation course'
        ]

        num_pipe = Pipeline(
            [
            ('imputer', SimpleImputer(strategy ='median')),
            ('scaler', StandardScaler())
            ]
        )

        cat_pipe = Pipeline(
            [
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('one_hot_encoding', OneHotEncoder()),
            ('scaler', StandardScaler(with_mean = False))
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ('num_pipeline', num_pipe, numerical_features),
                ('categorical_pipeline', cat_pipe, categorical_features)
            ]
        )

        return preprocessor



    def initiate_transformation(self, train_path, test_path):  # train and test path are obtained from data_ingestion
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        preprocessor_obj = self.get_transformer_obj()

        X_train_df = train_df.drop(columns = ['math score'], axis=1)
        y_train_df = train_df['math score']

        X_test_df = test_df.drop(columns = ['math score'], axis=1)
        y_test_df = test_df['math score']

        X_train_arr = preprocessor_obj.fit_transform(X_train_df)
        X_test_arr = preprocessor_obj.transform(X_test_df)


        train_arr = np.c_[X_train_arr, np.array(y_train_df)]
        test_arr = np.c_[X_test_arr, np.array(y_test_df)]

        print("Train columns:", train_df.columns.tolist())


        save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_obj)
        return(
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )





