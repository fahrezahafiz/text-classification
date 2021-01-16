import os
import re
import bz2
import nltk
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix

RAND_SEED = 121199
ROOT_DIR = './dataset'
corpus = []


def remove_stopwords(src):
    with open('stopwords.txt', 'r') as f:
        stopwords = f.readline()
    out = ' '.join(word for word in src.split() if word not in stopwords)
    return out


def lemmatize_text(src):
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in word_tokenize(src)]
    return ' '.join(lemmatized)


def read_documents(directory: str):
    # -- DATA --
    for root, dirs, _ in os.walk(directory):
        for category in dirs:
            print(category)
            for filename in os.listdir(os.path.join(root, category)):
                if filename.endswith(".txt"):
                    path_to_file = os.path.join(root, category, filename)
                    # print(path_to_file)
                    with open(path_to_file, 'r') as f:
                        doc = str(f.read().strip())
                        doc = doc.replace('\n', ' ')
                        doc = doc.encode('utf-8', errors='ignore').decode()
                        corpus.append([category, doc])

    df = pd.DataFrame(corpus, columns=['category', 'document'])
    print(df)
    print(df.head())
    return df


def preprocess(data: pd.DataFrame, test_size: float = 0.2):
    # -- PREPROCESS --
    # to lowercase
    data.loc[:, 'document'] = data.document.apply(lambda x: str.lower(x))
    # remove punctuation and numbers
    data.loc[:, 'document'] = data.document.apply(lambda x: " ".join(re.findall("[a-z]+", x)))
    # remove stopwords
    data.loc[:, 'document'] = data.document.apply(lambda x: remove_stopwords(x))
    # lemmatize
    data.loc[:, 'document'] = data.document.apply(lambda x: lemmatize_text(x))
    print(data.head())
    # train-test split 80:20
    X_train, X_test, y_train, y_test = train_test_split(data.document, data.category,
                                                        test_size=test_size, random_state=RAND_SEED)
    return X_train, X_test, y_train, y_test


def generate_tdm(x_train: pd.DataFrame, x_test: pd.DataFrame):
    # -- FEATURE EXTRACTION --
    # tf-idf
    vectorizer = TfidfVectorizer()
    train_tdm = vectorizer.fit_transform(x_train)
    train_tdm = pd.DataFrame(train_tdm.todense(), columns=vectorizer.get_feature_names())

    test_tdm = vectorizer.transform(x_test)
    test_tdm = pd.DataFrame(test_tdm.todense(), columns=vectorizer.get_feature_names())

    print(train_tdm.shape, test_tdm.shape)
    print(test_tdm)
    pkl_vectorizer = bz2.BZ2File('vectorizer.pickle', 'w')
    pickle.dump(vectorizer, pkl_vectorizer)

    return train_tdm, test_tdm


def tune_hyperparameters(model, params):
    tuned_model = RandomizedSearchCV(estimator=model,
                                     param_distributions=params,
                                     n_iter=50,
                                     cv=5,
                                     verbose=2,
                                     random_state=RAND_SEED,
                                     n_jobs=-1)

    return tuned_model


if __name__ == '__main__':
    # acquire data
    df = read_documents(ROOT_DIR)
    # preprocess
    X_train, X_test, y_train, y_test = preprocess(df, 0.2)
    # feature extraction
    train_tdm, test_tdm = generate_tdm(X_train, X_test)

    # hyperparameter tuning
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    # 20 iter
    best_1 = {'n_estimators': 400,
              'min_samples_split': 5,
              'min_samples_leaf': 1,
              'max_features': 'auto',
              'max_depth': 50,
              'bootstrap': False}
    # 25 iter
    best_2 = {'n_estimators': 1400,
              'min_samples_split': 5,
              'min_samples_leaf': 1,
              'max_features': 'auto',
              'max_depth': 40,
              'bootstrap': False}

    base_model = RandomForestClassifier()
    tuned_model = tune_hyperparameters(RandomForestClassifier(), random_grid)
    # tuned_model = RandomForestClassifier(**best_2)

    base_model.fit(train_tdm, y_train)
    tuned_model.fit(train_tdm, y_train)
    pkl_rfc = bz2.BZ2File('rfc.pickle', 'w')
    pickle.dump(tuned_model, pkl_rfc)
    print(f"Best params: {tuned_model.best_params_}")
    # testing
    base_pred = base_model.predict(test_tdm)
    tuned_pred = tuned_model.predict(test_tdm)
    base_cm = plot_confusion_matrix(base_model,
                                    test_tdm, y_test,
                                    display_labels=df.category.unique(),
                                    cmap=plt.cm.Blues,
                                    normalize='true')
    base_cm.ax_.set_title(f"Base model confusion matrix ({accuracy_score(y_test, base_pred) * 100:.2f})")
    tuned_cm = plot_confusion_matrix(tuned_model,
                                     test_tdm, y_test,
                                     display_labels=df.category.unique(),
                                     cmap=plt.cm.Blues,
                                     normalize='true')
    tuned_cm.ax_.set_title(f"Tuned model confusion matrix ({accuracy_score(y_test, tuned_pred) * 100:.2f})")
    plt.show()
