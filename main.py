from flask import Flask, jsonify, render_template, request, redirect, url_for, abort, session, Response
import pandas as pd
import numpy as np
import os
import spacy
import string
import re
import gensim
import json
from numba import jit, cuda
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import nltk

from nltk.util import ngrams
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, \
    f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import model_selection, metrics, svm, naive_bayes
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from collections import defaultdict
from collections import Counter

from tqdm import tqdm
import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
# from tensorflow.keras.initializers import Constant
# from tensorflow.keras.optimizers import Adam

import warnings

stop = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_lg")

warnings.simplefilter('ignore')

app = Flask(__name__, template_folder='templates', static_folder='static')
ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'This is your secret key to utilize session in Flask'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        if request.files:
            # upload file
            uploaded_file = request.files['file']
            data_filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
            session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        return '', 204


@app.route('/show_data')
def showData():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path, index_col=0)

    # pandas dataframe to html table flask
    uploaded_df_html = uploaded_df.head(10).to_html()
    charts = showCharts(uploaded_df, 'class')
    df = processText(uploaded_df, 'post').text_cleaning()
    bigram = showBigrams(df, 'class')
    labels = charts.getLabels()
    df.to_csv(data_file_path)

    return render_template('view.html', data=uploaded_df_html, distJSON=charts.plotDistribution(),
                           lengthJSON=charts.plotTextLength(), distplotJSON=charts.plotWordLength(),
                           bigramJSON=bigram.plot_bigrams(), labels=labels)


@app.route('/generateModels', methods=['GET', 'POST'])
def generateModels():
    if request.method == 'POST':
        data_file_path = session.get('uploaded_data_file_path', None)
        uploaded_df = pd.read_csv(data_file_path, index_col=0)
        data = request.json
        if data:
            check_box_labels = data['labels']
            check_box_models = data['models']
            if check_box_labels:
                svm_model = trainModels(uploaded_df, check_box_labels, check_box_models)
                model_pred = svm_model.train()
                return jsonify(model_pred)
        return jsonify({'error': 'Missing data!'})
        # return render_template('view.html', svm_pred=)


class showCharts:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.labels = df[target].unique()

    def plotDistribution(self):
        lengths = []
        for label in self.labels:
            lengths.append(self.df[self.df[self.target] == label].shape[0])

        d = {'labels': self.labels, 'samples': lengths}
        t = pd.DataFrame(d)
        fig = px.bar(t, x='labels', y='samples', color='labels', title='Data distribution')
        fig.update_layout(legend=dict(orientation="h",
                                      yanchor="bottom",
                                      y=1.02,
                                      xanchor="right",
                                      x=1
                                      ))
        distJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return distJSON

    def plotTextLength(self):
        colors = ["red", "green", "blue", "magenta"]

        fig = make_subplots(
            rows=1, cols=2,
            column_width=[0.5, 0.5],
            row_heights=[1],
            subplot_titles=('Characters in text', 'Words in text'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}]]
        )

        for label in self.labels:
            fig.add_trace(go.Histogram(
                x=self.df[self.df[self.target] == label]['post'].str.len(),
                name=label,
                marker=dict(color=colors[np.where(self.labels == label)[0][0]]),
                legendgroup="group"
            ),
                row=1, col=1
            )

            fig.add_trace(go.Histogram(
                x=self.df[self.df[self.target] == label]['post'].str.split().map(lambda x: len(x)),
                name=label,
                marker=dict(color=colors[np.where(self.labels == label)[0][0]]),
                legendgroup="group",
                showlegend=False),
                row=1, col=2
            )

        fig.update_layout(barmode='overlay',
                          legend=dict(orientation="h",
                                      yanchor="bottom",
                                      y=1.02,
                                      xanchor="right",
                                      x=1
                                      ))

        fig.update_traces(opacity=0.75)

        fig.update_xaxes(range=[0, 200], row=1, col=1)
        fig.update_xaxes(range=[0, 50], row=1, col=2)
        fig.update_xaxes(title_text='length', row=1, col=1)
        fig.update_xaxes(title_text='length', row=1, col=2)
        fig.update_yaxes(title_text='sample number', row=1, col=1)
        fig.update_yaxes(title_text='sample number', row=1, col=2)

        lengthJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return lengthJSON

    # @jit(target="cuda")
    def plotWordLength(self):
        hist_data = []

        for label in self.labels:
            word = self.df[self.df[self.target] == label]['post'].str.split().apply(lambda x: [len(i) for i in x])
            hist_data.append(word.map(lambda x: np.mean(x)))

        fig = create_distplot(hist_data, self.labels, bin_size=0.2)
        fig.update_xaxes(range=[0, 10])
        fig.update_layout(title_text='Word length', legend=dict(orientation="h",
                                                                yanchor="bottom",
                                                                y=1.02,
                                                                xanchor="right",
                                                                x=1
                                                                ))

        distplotJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return distplotJSON

    def getLabels(self):
        j = pd.Series(self.labels).to_json(orient='values')
        return j


class processText:
    def __init__(self, df, text):
        self.df = df
        self.text = text

    @staticmethod
    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    @staticmethod
    def remove_html(text):
        html = re.compile(r'<.*?>')
        return html.sub(r'', text)

    @staticmethod
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def remove_punct(text):
        for punc in list(string.punctuation):
            if punc in text:
                text = text.replace(punc, '')
        return text.strip()
        # table = str.maketrans('', ' ', string.punctuation)
        # return text.translate(table)

    @staticmethod
    def remove_rt(text):
        return re.sub(r'\brt\b', '', text)

    @staticmethod
    def stopword(text):
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in stop]
        return " ".join(tokens_without_sw)

    def text_cleaning(self):
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_URL(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_html(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_emoji(x))
        # self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_punct(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_rt(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.stopword(x))
        self.df.reset_index(inplace=True, drop=True)

        return self.df


class showBigrams:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.labels = df[target].unique()

    @staticmethod
    def get_top_bi_grams(corpus, n=None):
        vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def plot_bigrams(self):
        bigrams = self.get_top_bi_grams(self.df['post'], 15)
        y, x = map(list, zip(*bigrams))
        d = pd.DataFrame({'bi-gram': y, 'frequency': x})
        fig = px.bar(d, x='frequency', y='bi-gram', color='bi-gram', orientation='h')
        fig.update_layout(title_text='top bi-grams', showlegend=False)

        bigramsJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return bigramsJSON


class trainModels:
    def __init__(self, df, labels, models):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.df = df
        self.labels = labels
        self.models = models

    def processText(self):
        self.df = self.df[self.df['class'].isin(self.labels) == True]
        self.df.reset_index(inplace=True, drop=True)
        self.df['post'].dropna(inplace=True)
        self.df['post'] = [entry.lower() for entry in self.df['post']]
        self.df['post'] = [word_tokenize(entry) for entry in self.df['post']]
        tag_map = defaultdict(lambda: wordnet.NOUN)
        tag_map['J'] = wordnet.ADJ
        tag_map['V'] = wordnet.VERB
        tag_map['R'] = wordnet.ADV
        for i, entry in enumerate(self.df['post']):
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                    Final_words.append(word_Final)
            self.df.loc[i, 'text_final'] = str(Final_words)

    def prepareData(self, text):
        self.processText()
        X_train, X_test, y_train, y_test = train_test_split(self.df[text],
                                                            self.df['class'],
                                                            test_size=0.2,
                                                            random_state=42)

        Encoder = LabelEncoder()
        self.y_train = Encoder.fit_transform(y_train)
        self.y_test = Encoder.fit_transform(y_test)
        tfidfVec = TfidfVectorizer(max_features=5000)
        tfidfVec.fit(self.df[text])

        self.X_train = tfidfVec.transform(X_train)
        self.X_test = tfidfVec.transform(X_test)

    def trainSVM(self):
        model_svm = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        model_svm.fit(self.X_train, self.y_train)

        pred_svm = model_svm.predict(self.X_test)

        return f'Accuracy Score SVM: "{round(accuracy_score(self.y_test, pred_svm) * 100,2)}" \n', \
               f'Precision Score SVM: "{round(precision_score(self.y_test, pred_svm) * 100,2)}" \n', \
               f'F1 Score SVM: "{round(f1_score(self.y_test, pred_svm, average="binary") * 100,2)}" \n'

    def trainNaiveBayes(self):
        model_naive = naive_bayes.MultinomialNB()
        model_naive.fit(self.X_train, self.y_train)

        pred_svm = model_naive.predict(self.X_test)

        return f'Accuracy Score NB: "{round(accuracy_score(self.y_test, pred_svm) * 100,2)}" \n', \
               f'Precision Score NB: "{round(precision_score(self.y_test, pred_svm) * 100,2)}" \n', \
               f'F1 Score NB: "{round(f1_score(self.y_test, pred_svm, average="binary") * 100,2)}" \n'

    def train(self):
        self.prepareData('text_final')
        output = []
        if 'SVM' in self.models:
            output.append(self.trainSVM())

        if 'Naive' in self.models:
            output.append(self.trainNaiveBayes())

        return output


if __name__ == "__main__":
    app.run(debug=True)
