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
train_X = train.drop(['Unnamed: 0', 'time', 'Valencia_wind_deg', 'Seville_pressure', 'Valencia_pressure'], axis=1, inplace=True)

# Fit model
"""lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)
"""
randomforest =RandomForestRegressor()
print ("Training Model...")
randomforest = randomforest.fit(train_X,train_y)  

# Pickle model for use within our API
save_path = '../assets/trained-models/randomforest_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(randomforest, open(save_path,'wb'))
