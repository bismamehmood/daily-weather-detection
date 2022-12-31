from fastapi import FastAPI
from pydantic import BaseModel
from model import WeatherPrediction

app = FastAPI()
weather_model = None
class WeatherData(BaseModel):
    tmin: list[float]
    tmax: list[float]
    wdir: list[float]
    wspd: list[float]
    pres: list[float]

@app.on_event('startup')
async def setup_model():
    global weather_model
    weather_model = WeatherPrediction()
    weather_model.init_linear_model()
    # weather_model.evaluate_model()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(data: WeatherData):
    return weather_model.make_prediction(data.dict())
