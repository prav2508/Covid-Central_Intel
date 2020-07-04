#importing required libs
import pandas as pd
from pandas.io.json import json_normalize 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle


#fetching and exporting the json data
data = pd.read_json('https://corona-api.com/timeline')
data_frame = pd.json_normalize(data['data'])
data_frame = data_frame[['date','confirmed']]
data_frame.to_excel('data_frame_csv\covid_cases.xlsx', encoding='utf-8',index=False)


#fetching data for training
data = pd.read_excel('covid_cases_validated.xlsx')

data['date'] = pd.to_datetime(data['date'],format = '%Y-%m-%d', errors = 'coerce')
assert data.date.isnull().sum() == 0, 'missing ScheduledDay dates'
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
X_train = data.iloc[:,-3:].iloc[::-1].values
y_train = data.iloc[:, 1:-3].iloc[::-1].values


#modelling
poly_feat = PolynomialFeatures(degree=2)
x_poly = poly_feat.fit_transform(X_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)

#pickling the model
pkl_filename = "covid_prediction_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(lin_reg2, file)

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

predicted = pickle_model.predict(poly_feat.fit_transform([[2020,8,2]]))
print(int(predicted.ravel()[0]))

#predicting
y_preds = lin_reg2.predict(poly_feat.fit_transform([[2020,7,15]]))
print(int(y_preds.ravel()[0]))

