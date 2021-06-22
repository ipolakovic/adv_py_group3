from flask import Flask, request
from ie_bike_model.model import train_and_persist, predict

app = Flask(__name__)

@app.route('/train')
def do_train_and_persist():
    return train_and_persist()

@app.route('/predict')
def do_predict():
    """
    INPUT: 
    http://0.0.0.0:5000/predict?date=2012-11-01&hour=10&weather_situation=clear&temperature=0.3&feeling_temperature=0.31&humidity=0.8&windspeed=0.0
    OUTPUT: 
    call to predict with parameters in format expecting; should return prediction value
    """
    #extract and map param values from URL string to variables; pass to function
    dteday = request.args['date']
    hr = request.args['hour']
    weathersit = request.args['weather_situation']
    temp = request.args['temperature']
    atemp = request.args['feeling_temperature']
    hum = request.args['humidity']
    windspeed = request.args['windspeed']

    return predict(dteday, hr, weathersit, temp, atemp, hum, windspeed)
