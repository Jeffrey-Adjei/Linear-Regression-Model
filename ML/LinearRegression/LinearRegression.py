import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklear.linear_model import LinearRegression 

class LinearRegressionModel:
    def __init__(self, weight, bias, alpha):
        self.weight = 0
        self.bias = 0
        self.alpha = 0.01

    def load_dataset(self):
        data = pd.read_csv('NVIDIA Historical Market Data.csv')
        x = data.iloc[: , :-1]
        y = data[data.columns[:-1]]
        return x, y

    def split_dataset(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) 
        return x_train, x_test, y_train, y_test

    def train_linear_model(x_train, y_train):
        model = 
