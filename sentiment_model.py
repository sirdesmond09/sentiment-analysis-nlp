import pandas as pd
import numpy as np
from html import unescape
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import spacy
from spacy.lang.en import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
import gzip
import dill
import time


nlp = spacy.load('en', disable = ['ner', 'parser', 'tagger'])
STOP_WORD_lemma = [word.lemma_ for word in nlp(" ".join(list(STOP_WORDS)))]
STOP_WORD_lemma = set(STOP_WORD_lemma).union(',', '.', ';')


class OnlinePipeline(Pipeline):

    def partial_fit(self, X, y):
        try:
            Xt = X.copy()

        except AttributeError:
            Xt = X

        for _, est in self.steps:
            if hasattr(est, "partial_fit") and hasattr(est, "predict"):
                est.partial_fit(Xt, y)

            if hasattr(est, "transform"):
                Xt = est.transform(Xt)

        return self


def fit_model(func):

    def wrapper(*args, **kwargs):
        t_0 = time.time()
        model = func()
        model.fit(X_train, y_train)

        t_elapsed = time.time() - t_0

        print("training time: {:g}".format(t_elapsed))
        print("training accuracy: {:g}".format(model.score(X_train, y_train)))
        print("testing accuracy: {:g}".format(model.score(X_test, y_test)))

        return model

    return wrapper


def preprocessor(doc):
    '''Convert html entities in the doc to the correct character  and return a lower case of the doc'''
    return unescape(doc).lower()

def lemmatizer(doc):
    """Takes a document from a corpos and returns the lema(dictionary) words."""
    return [word.lemma_ for word in nlp(doc)]


@fit_model
def online_model():
    vectorizer = HashingVectorizer(preprocessor = preprocessor,
                            tokenizer=lemmatizer,
                             alternate_sign = False,
                            ngram_range = (1, 2),
                            stop_words=STOP_WORD_lemma)

    clf = SGDClassifier(loss = 'log', max_iter=5)

    pipe = OnlinePipeline([('vectorizer', vectorizer),
                            ('classifier', clf)])

    return pipe

@fit_model
def construct_model():
    vectorizer = TfidfVectorizer(preprocessor = preprocessor,
                                tokenizer=lemmatizer,
                                ngram_range = (1, 2),
                                stop_words=STOP_WORD_lemma)
    clf = MultinomialNB()

    pipe = Pipeline([('vectorizer', vectorizer), 
                    ('classifier', clf)])

    return pipe


def serialize_model():
    model = construct_model()

    with gzip.open("sentiment_model.dill.gz", 'wb') as f:
        dill.dump(model, f, recurse=True)



if __name__ == '__main__':
    df = pd.read_csv('Sentiment Analysis Dataset.csv', error_bad_lines = False)
    X = df['SentimentText']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

    model = construct_model() 