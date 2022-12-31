import json
from flask import Flask, request, jsonify
from model import WeatherPrediction

app = Flask(__name__)


weather_model = WeatherPrediction()
weather_model.init_linear_model()
@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data)
    prediction = weather_model.make_prediction(data.dict())
    return jsonify(prediction)