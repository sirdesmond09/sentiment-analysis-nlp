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

    proba = model.predict_proba([tweet])[0, 1]

    #positive class = class 1
    #negative class = class 0

    return "Positive sentiment {}".format(proba)
if __name__ == '__main__':
    app.run()