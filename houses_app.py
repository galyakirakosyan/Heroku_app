from flask import Flask, request
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

app = Flask(__name__)

filename = 'regr_houses.pickle'
loaded_model = pickle.load(open(filename, 'rb'))

@app.route("/")
def hello():
    return "type predict?full_sq=100&life_sq=70&num_room=4&kitch_sq=20&build_year=1990&floor=5&max_floor=10"

@app.route("/predict")

def house_price():
    full_sq = float(request.args.get("full_sq"))
    life_sq = float(request.args.get("life_sq"))
    num_room = float(request.args.get("num_room"))
    kitch_sq = float(request.args.get("kitch_sq"))
    build_year = float(request.args.get("buid_year"))
    floor = float(request.args.get("floor"))
    max_floor = float(request.args.get("max_floor"))
    result = loaded_model.predict([[full_sq, life_sq, num_room, kitch_sq, build_year, floor, max_floor]])
    return str(result)
    
if __name__=="__name__":
    app.run(debug = True)
