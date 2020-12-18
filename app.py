from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
import nltk
import bz2
import re

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        print('get method')
        return render_template('index.jinja')
    elif request.method == 'POST':
        doc = None
        if bool(request.form):
            doc = request.form['text-input']
        elif bool(request.files):
            file = request.files['file-input']
            doc = file.read().decode('utf-8')

        if doc is not None:
            vectorizer = None
            rfc = None
            zipped_vec = bz2.BZ2File('vectorizer.pickle')
            vectorizer = pickle.load(zipped_vec)
            print(vectorizer)
            zipped_rfc = bz2.BZ2File('rfc.pickle')
            rfc = pickle.load(zipped_rfc)
            # feature extraction
            doc = preprocess_input(doc)
            test_feature = vectorizer.transform([doc])
            test_feature = pd.DataFrame(test_feature.todense(), columns=vectorizer.get_feature_names())
            print(f"test_feature:\n{test_feature}")
            # prediction
            pred = rfc.predict(test_feature)[0]
            pred = pred
            print(f"prediction: {pred}")
        
        
        return render_template('index.jinja', prediction=pred)


def preprocess_input(src):
    # lowercase
    out = str.lower(src)
    # remove punctuation and numbers
    out = " ".join(re.findall("[a-z]+", out))
    # remove stopwords
    with open('stopwords.txt', 'r') as f:
        stopwords = f.readline()
    out = " ".join(word for word in out.split() if word not in stopwords)
    # lemmatize
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in word_tokenize(out)]
    out = " ".join(lemmatized)
    print(f"preprocessed: {out}")
    return out
