"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

train_y = train[['load_shortfall_3h']]
train_X = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

# Fit model
"""lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)
"""
randomfor =RandomForestRegressor(max_depth=4, max_features="sqrt")
print ("Training Model...")
randomfor = randomfor.fit(train_X,train_y)  

# Pickle model for use within our API
save_path = '../assets/trained-models/randomfor_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(randomfor, open(save_path,'wb'))
