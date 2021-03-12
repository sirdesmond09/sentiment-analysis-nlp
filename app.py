from flask import Flask, request, render_template, redirect
import dill
import gzip

from requests.sessions import Request

app = Flask(__name__)

@app.route('/')
def main():
    return redirect('/index')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return "This page is all about my ML model"



@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    if request.method == 'GET':

        tweet = request.args.get("tweet")
    else:
        tweet = request.form['text']
        
    with gzip.open('sentiment_model.dill.gz', "rb") as f:
        model = dill.load(f)

    proba = round(model.predict_proba([tweet])[0, 1] * 100, 2)

    p = model.predict([tweet])

    if p == 0:
        result = "negative"
    elif p == 1:
        result = 'positive'
    value = [proba, result]
    #positive class = class 1
    #negative class = class 0

    return render_template('result.html', value=value)


if __name__ == '__main__':
    app.run()