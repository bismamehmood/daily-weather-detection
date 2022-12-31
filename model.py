from math import sqrt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


class WeatherPrediction:
    file_path = 'daily_weather_data.csv'

    def __init__(self):
        self.df = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.linear_regression = None
        self.laod_data()
        self.clean_data()

    def laod_data(self):
        self.df = pd.read_csv(self.file_path, date_parser=['date'])
        print(self.df.head())
    

    def clean_data(self):
        self.df = self.df.drop_duplicates(keep='first')
        self.df.sort_values(by=['date'], inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.df = self.df.drop(columns=['country','city', 'Latitude', 'Longitude', 'date'])
        self.df = self.df[self.df['tavg'].notna()]
        print("="*10, "Status", "="*10)
        print(self.df.isnull().sum(axis=0) * 100/len(self.df))
        self.df = self.prepare_data(self.df)


    def prepare_data(self, data_frame):
        data_frame = data_frame[data_frame['wdir'].notna()]
        data_frame = data_frame[data_frame['pres'].notna()]
        data_frame = data_frame[data_frame['wspd'].notna()]
        data_frame = data_frame[data_frame['tmax'] <= 60]
        data_frame = data_frame[data_frame['wspd'] <= 70]
        data_frame = data_frame[data_frame['pres'] <= 1050]
        data_frame = data_frame[data_frame['pres'] > 950]
        data_frame.reset_index(inplace=True, drop=True)
        print("="*10, "Dataframe description", "="*10)
        print(data_frame.describe(include='all'))
        print("="*20)
        data_frame = self.transform_data(data_frame)
        return data_frame

    def transform_data(self, data_frame):
        # squre root transformation for 'wspd'
        sqrt_transformer = FunctionTransformer(np.sqrt, validate=True)
        data_transformed = sqrt_transformer.transform(data_frame[['wspd']])
        data_frame['t_wspd'] = data_transformed
        # power transformation with power of 2 for 'pres'
        power_transformer = FunctionTransformer(lambda x: x**2, validate=True)
        data_transformed = power_transformer.transform(data_frame[['pres']])
        data_frame['t-pres'] = data_transformed
        return data_frame

    def generate_train_and_test_data(self):
        df_data = self.df.drop('tavg', axis=1)
        df_target = pd.DataFrame(self.df['tavg'], columns=['tavg'])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            df_data, df_target, test_size=0.2, random_state=42)


    def init_linear_model(self):
        self.generate_train_and_test_data()
        self.linear_regression = linear_model.LinearRegression()
        self.linear_regression.fit(self.x_train, self.y_train)
    

    def evaluate_model(self):
        prediction = self.linear_regression.predict(self.x_test)
        
        # Mean Squared Error
        mse = mean_squared_error(self.y_test, prediction)
        mse = round(mse, 4)
        print('Mean squared error(Train) :', round(mse, 4))

        # Root Mean Squared Error
        rmsq = sqrt(mean_squared_error(self.y_test, prediction))
        rmsq = round(rmsq, 4)
        print('Root mean squared error(Train) :', round(rmsq, 4))

        # Accuracy
        accuracy = self.linear_regression.score(self.x_test, prediction)
        accuracy = round(accuracy * 100, 4)
        print('Explained variance of the predictions(Train) :', round(accuracy * 100, 4))

        return mse, rmsq, accuracy
    
    def make_prediction(self, data):
        df = pd.DataFrame(data)
        df = self.transform_data(df)
        prediction = self.linear_regression.predict(df)
        return prediction.tolist()


if __name__ == '__main__':
    obj = WeatherPrediction()
    obj.init_linear_model()
    obj.evaluate_model()
    # , , , , , 3.478505426185217, 1038564.81
    test_data = {
    "tmin": [21.6],
    "tmax": [25.6],
    "wdir": [312.0],
    "wspd": [12.1],
    "pres": [1019.1],
    }

    print(obj.make_prediction(pd.DataFrame(test_data)))