#Student Name: Joshua Salas
#Date: October 14, 2023

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load the datasets using pandas
X_train = pd.read_csv('lab2_dataset-2/q3/X_train_q3.csv')
y_train = pd.read_csv('lab2_dataset-2/q3/y_train_q3.csv')
X_test = pd.read_csv('lab2_dataset-2/q3/X_test_q3.csv')

def linear_regression_equation():
    mse = 0

    #Creating a weight_vector where all values are 1.
    weight_vector = [1 for i in range(len(X_train.columns))]
    dot_product = []

    #getting the dot_product of each row in X_train.csv
    for row in range(len(X_train)):
        #for each element in row we multiply by corresponding value in weight_vector.
        #then we sum all the results.
        d = sum([X_train[row,i] * weight_vector[i] for i in range(len(row))])
        dot_product.append(d)
            
    #converting y_train.csv to list so we can use imported function mean_squared_error().
    head = y_train.columns
    actual = list(y_train[head[0]])
    mse = mean_squared_error(actual,dot_product,)

    print("MSE using library: ", mse) 
    
def linear_regression_library():
    # 3. Implement linear regression using scikit-learn

    LR = LinearRegression()
    LR.fit(X_train,y_train)

    # 4. Predict the output for the test dataset
    y_pred = LR.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred, columns=['Chance of Admit']).to_csv("y_predict_q3.csv", index=False)
    
linear_regression_equation()
linear_regression_library()