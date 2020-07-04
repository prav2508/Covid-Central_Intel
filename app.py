import sys
sys.path.append("/home/lkm/Folder/project/app/myvenv/python3.4/site-packages")
from flask import Flask
from sklearn.preprocessing import PolynomialFeatures
import pickle
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

result={}
pkl_filename = 'covid_prediction_model.pkl'
poly_feat = PolynomialFeatures(degree=2)

@app.route('/')
@cross_origin()
def welcome():
    return "<h1>Welcome to covid central intel</h1><br/><br/> <h4>Add the date as string to the url to get covid count prediction!!!</h4>"


@app.route('/predict/<date>')
@cross_origin()
def predict(date):
    selected_date = date.split('-')
    print(selected_date)
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    predicted = pickle_model.predict(poly_feat.fit_transform([[int(selected_date[0]),int(selected_date[1]),int(selected_date[2])]]))
    #predicted = pickle_model.predict(poly_feat.fit_transform([[2020,7,5]]))
    count = int(predicted.ravel()[0])
    json_data = json.loads('{"total_count":'+str(count)+'}')
    return json_data

if __name__ == '__main__':
    app.run(debug=True)