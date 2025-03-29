import pandas as pd 
from sklearn.model_selection import train_test_split
from sklear.linear_model import LinearRegression 
from sklearn.metrics import r2, mean_square_error

class LinearRegressionModel:
    def __init__(self, weight, bias, alpha, epochs):
        self.weight = 0.0
        self.bias = 0.0
        self.alpha = 0.0001
        self.epochs = 1500

    def load_dataset(self):
        data = pd.read_csv('rawdata.csv')
        x = data['X']
        y = data['Y']
        return x, y

    def split_dataset(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)          
        return x_train, x_test, y_train, y_test

    def train_linear_model(x_train, y_train):
        model = LinearRegression()
        model.fit(x_train, y_test)
        return model
    
    def model_predictions(model, x_test):
        y_pred = model.predict(x_test)
        return y_pred

    def evaluate_regression_model(y_pred):
        r2_score = r2(y_test, y_pred)
        mse = mean_square_error(y_test, y_pred)
        print(f"coefficient of determination: ", {r2_score})
        print(f"mean square error: ", {mse})


    def gradient_descent_optimisation(m_now, b_now, points, self.alpha):
        m_gradient = 0 
        b_gradient = 0 

        n = len(points)

        for i in range(n):
            x = points.iloc[i].X
            y = points.iloc[i].Y

            m_gradient += -(2/n) * x * (y - (self.weight * x + self.bias)) 
            b_gradient += -(2/n) * (y - (self.weight * x + self.bias)) 
        
        self.weight = self.weight - m_gradient * self.alpha
        self.bias = self.bias - b_gradient * self.alpha

        return self.weight, self.bias

    def plot_regression_line():
        pass



if "__name__" == '__main__':
    main()
