from flask import Flask
from flask import request
from flask import render_template

from io import BytesIO
import base64

import matplotlib.pyplot as plt

from box_office_revenue_predictor import get_scores, get_plots

app = Flask(__name__)

@app.route("/")
def main_page():
    return render_template('home.html')


@app.route("/visualize")
def visualize():
    
    percentage = request.args.get('percentage')
    
    if percentage is None:
        percentage = 100
    else:
        percentage = int(percentage)
    
    print(type(percentage), percentage)
    items = get_plots(percentage)
    scores = get_scores()
    
    for score in scores:
        score[1] = round(score[1], 5)
        score[2] = round(score[2], 5)
        if score[3] != '-':
            score[3] = round(score[3], 5)
    
    return render_template('visualizations.html', plots=items, scores=scores)


@app.route('/predict')
def predict():
    return render_template('predict.html')