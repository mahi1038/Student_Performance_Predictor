import pandas as pd
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        model_path = 'E:\environments\Student_Performance_Project\dataset\model.pkl'
        preprocessor_path = 'E:\environments\Student_Performance_Project\dataset\preprocessor.pkl'
        model = load_object(file_path = model_path)
        preprocessor = load_object(file_path = preprocessor_path)
        data_scaled = preprocessor.transform(features)
        predicted_data = model.predict(data_scaled)
        return predicted_data

class CustomData:
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education: str,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self):

        input = {
            "gender": [self.gender],
            "race ethnicity": [self.race_ethnicity],
            "parental level of education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test preparation course": [self.test_preparation_course],
            "reading score": [self.reading_score],
            "writing score": [self.writing_score]
        }

        return pd.DataFrame(input)


        

