# imports
import secrets
from statistics import mean

from flask import Flask, jsonify, render_template, request, session, send_file, flash
from transformers import BertTokenizer, BertModel

from flask_session import Session
import pandas as pd
import numpy as np
import torch
import os
import spacy
import string
import re
import json
import pickle
import shutil
import mkl
import contractions

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot

from typing import Callable, List, Optional, Tuple
from zeugma.embeddings import EmbeddingTransformer

import nltk
from nltk import pos_tag, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import svm, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from collections import defaultdict

import warnings

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
stop = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_lg")

warnings.simplefilter('ignore')

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = os.path.join('static', 'uploads')
MODEL_FOLDER = os.path.join('static', 'models')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

Session(app)

Encoder = LabelEncoder()
PredictorScaler = MinMaxScaler()


# flask index route
@app.route('/')
def index():
    """
    :return:
    """
    session.clear()
    return render_template('index.html')


# upload and safe .csv file in directory
@app.route('/', methods=['POST'])
def upload_file():
    """

    :return:
    """
    if request.form:
        uploaded_file = request.files['file_main']
        uploaded_test_file = request.files['file_test']

        file_encoding = request.form['file_encoding']
        file_seperator = request.form['seperator']
        file_header = request.form['header']
        file_quote = request.form['quote']
        file_index = request.form['index']

        if not file_seperator:
            file_seperator = ','

        if 't' in file_seperator:
            file_seperator = '\t'

        if 's+' in file_seperator:
            file_seperator = '\s+'

        if file_header == 'yes':
            file_header = 1
        else:
            file_header = 'infer'

        if file_index == 'None':
            file_index = None
        else:
            file_index = int(file_index)

        data_filename = secure_filename(uploaded_file.filename)
        data_test_filename = secure_filename(uploaded_test_file.filename)

        try:
            uploaded_file = pd.read_csv(uploaded_file, encoding=file_encoding, sep=file_seperator, quotechar=file_quote,
                                        header=file_header, index_col=file_index)
        except FileNotFoundError:
            uploaded_file = pd.read_csv(uploaded_file, encoding=file_encoding, sep=None)
        except:
            flash('Could not read file.')
            return render_template('index.html')

        if uploaded_test_file:
            try:
                uploaded_test_file = pd.read_csv(uploaded_test_file, encoding=file_encoding, sep=file_seperator,
                                                 quotechar=file_quote, header=file_header, index_col=file_index)
            except FileNotFoundError:
                uploaded_test_file = pd.read_csv(uploaded_test_file, encoding=file_encoding, sep=None)
            except:
                flash('Could not read file.')
                return render_template('index.html')

            uploaded_test_file.dropna(inplace=True)
            uploaded_test_file.reset_index(drop=True, inplace=True)
            uploaded_test_file.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], data_test_filename))
            session['uploaded_data_test_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_test_filename)

        uploaded_file.dropna(inplace=True)
        uploaded_file.reset_index(drop=True, inplace=True)
        uploaded_file.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

        session.modified = True

    return '', 204


# redirect to /select, return columns and head of data
@app.route('/select', methods=['POST'])
def select():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    # pandas dataframe to html table flask
    uploaded_df_html = uploaded_df.head(10).to_html()
    charts = showCharts(uploaded_df)
    features = charts.getFeatures()

    return render_template('select.html', data=uploaded_df_html, features=features)


# ajax route, returns charts and labels
@app.route('/showData', methods=['POST'])
def showData():
    data_file_path = session.get('uploaded_data_file_path', None)

    uploaded_df = pd.read_csv(data_file_path, index_col=0)

    data = request.json
    if data:
        check_box_target = data['target'][0]
        check_box_text = data['text'][0]
        check_box_chart = data['chart']
        session['target'] = check_box_target
        session['text'] = check_box_text
        session.modified = True
        if 'df' not in session:
            df = processText(check_box_text, uploaded_df).text_cleaning_df()
            df.to_csv(data_file_path)
            session['df'] = True
        else:
            df = uploaded_df

        charts = showCharts(uploaded_df, check_box_text, check_box_target)
        out = {}

        if check_box_chart:
            for chart in check_box_chart:
                if chart in session:
                    chart_output = session.get(chart)
                else:
                    if chart == 'Bi-Grams':
                        chart_output = showBigrams(df, check_box_text, check_box_target).plot_bigrams()
                    if chart == 'Distribution':
                        chart_output = charts.plotDistribution()
                    if chart == 'Text Length':
                        chart_output = charts.plotTextLength()
                    if chart == 'Word Length':
                        chart_output = charts.plotWordLength()

                session[chart] = chart_output
                out.update({chart: chart_output})

        if 'labels' in session:
            out.update({'labels': session.get('labels')})
        else:
            labels = charts.getLabels()
            out.update({'labels': labels})

        return out
    return jsonify({'error': 'Missing data!'})


# redirect to view.html, calculate selected models with options
@app.route('/view', methods=['POST'])
def view():
    data_file_path = session.get('uploaded_data_file_path', None)
    uploaded_df = pd.read_csv(data_file_path, index_col=0)

    check_box_labels = request.form.getlist('label')
    check_box_models = request.form.getlist('model')
    check_box_options = request.form.getlist('option')
    check_box_class_weight = request.form.get('class_weight')
    check_box_cross_val = request.form.get('cross_val')
    check_box_average = request.form.getlist('average')[0]
    slider_split = int(request.form.get('slider')) / 100
    pos_label = 1

    text = session.get('text', None)
    target = session.get('target', None)

    for i in range(len(check_box_labels)):
        if "pos_label" in check_box_labels[i]:
            check_box_labels[i] = check_box_labels[i].replace(' pos_label', '')
            if pos_label != 1:
                pos_label = 1
                break
            pos_label = check_box_labels[i]

    if check_box_class_weight == 'on':
        check_box_class_weight = True
    else:
        check_box_class_weight = False

    if check_box_cross_val == 'on':
        check_box_cross_val = True
    else:
        check_box_cross_val = False

    models = trainModels(uploaded_df, text, target, check_box_labels, check_box_models, check_box_class_weight,
                         check_box_average, slider_split, check_box_cross_val,pos_label, check_box_options)

    model_pred, model_pred_cross_val = models.plotOutput()
    if model_pred_cross_val:
        model_pred_cross_val = model_pred_cross_val.to_html()
    else:
        model_pred_cross_val = ""

    return render_template('view.html', model=model_pred.to_html(),
                           model_pred_cross_val=model_pred_cross_val,
                           labels=check_box_labels)


# download trained model as pipelines with pickle
@app.route('/download')
def download_file():
    path = shutil.make_archive('models', 'zip', os.path.join(app.config['MODEL_FOLDER']))
    return send_file(path, as_attachment=True)


# returns predicted label of the test input text
@app.route('/test_input', methods=['POST'])
def predict_text():
    data = request.json['text']
    data = processText(data).text_cleaning_sentence()
    models_file = session.get('trained_models', None)

    out = {}
    for file_name in models_file:
        model = pickle.load(open(os.path.join(app.config['MODEL_FOLDER'], file_name), 'rb'))
        prediction = ' '.join(Encoder.inverse_transform(model.predict(data)))
        file_name = file_name.replace('.pkl', '')
        out.update({file_name: prediction})
    df = pd.DataFrame(list(out.items()))
    df.columns = ['Model', 'Prediction']

    return df.to_json(orient='records')


# calculate charts
class showCharts:
    def __init__(self, df, text=0, target=0):
        """

        :param df: pandas Dataframe
        :param text: text column in dataframe
        :param target: target column in dataframe
        """
        self.df = df
        self.color = ['#284B63', '#4D194D', '#d62828', '#7a0045', 'red', 'green', 'blue', 'yellow']
        if target != 0:
            self.target = target
            self.labels = df[target].unique()
        if text != 0:
            self.text = text

    def plotDistribution(self):
        """
        :return: data distribution per unique label of dataset as plotly json object
        """
        lengths = []
        percentage = []

        for label in self.labels:
            length = self.df[self.df[self.target] == label].shape[0]
            lengths.append(length)
            percentage.append(f'{round((length / self.df.shape[0]) * 100, 2)}%')

        d = {'labels': [str(x) for x in self.labels], 'samples': lengths, 'percentage': percentage}
        t = pd.DataFrame(d)
        fig = px.bar(t, x='labels', y='samples', color='labels', title='Data distribution',
                     color_discrete_sequence=self.color, hover_data=['labels', 'percentage'])
        fig.update_layout(legend=dict(orientation="h",
                                      yanchor="bottom",
                                      y=1.02,
                                      xanchor="right",
                                      x=1
                                      ),
                          paper_bgcolor='#FFFFFF',
                          plot_bgcolor='#D9D9D9',
                          font=dict(color='#284B63', family="Lucida Console", size=20))
        distJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return distJSON

    def plotTextLength(self):
        """
        :return: average text length of the dataset (total amount of characters), as plotly json object
        """

        fig = make_subplots(
            rows=1, cols=2,
            column_width=[0.5, 0.5],
            row_heights=[1],
            subplot_titles=('Characters in text', 'Words in text'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}]]
        )

        for label in self.labels:
            fig.add_trace(go.Histogram(
                x=self.df[self.df[self.target] == label][self.text].str.len(),
                name=str(label),
                marker=dict(color=self.color[np.where(self.labels == label)[0][0]]),
                legendgroup="group"
            ),
                row=1, col=1
            )

            fig.add_trace(go.Histogram(
                x=self.df[self.df[self.target] == label][self.text].str.split().map(lambda x: len(x)),
                name=str(label),
                marker=dict(color=self.color[np.where(self.labels == label)[0][0]]),
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
                                      ),
                          paper_bgcolor='#FFFFFF',
                          plot_bgcolor='#D9D9D9',
                          font=dict(color='#284B63', family="Lucida Console", size=20))

        fig.update_traces(opacity=0.65)

        fig.update_xaxes(row=1, col=1)
        fig.update_xaxes(row=1, col=2)
        fig.update_xaxes(title_text='length', row=1, col=1)
        fig.update_xaxes(title_text='length', row=1, col=2)
        fig.update_yaxes(title_text='sample number', row=1, col=1)
        fig.update_yaxes(title_text='sample number', row=1, col=2)

        lengthJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return lengthJSON

    def plotWordLength(self):
        """
        :return: average word length (average amount of characters per word), as plotly json object
        """
        hist_data = []
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        for label in self.labels:
            word = self.df[self.df[self.target] == label][self.text].str.split().apply(lambda x: [len(i) for i in x])
            hist_data.append(word.map(lambda x: np.mean(x)))

        fig = create_distplot(hist_data, [str(x) for x in self.labels], bin_size=0.2, colors=self.color)
        fig.update_xaxes(title_text='length', range=[0, 10])
        fig.update_layout(title_text='Word length', legend=dict(orientation="h",
                                                                yanchor="bottom",
                                                                y=1.02,
                                                                xanchor="right",
                                                                x=1
                                                                ),
                          paper_bgcolor='#FFFFFF',
                          plot_bgcolor='#D9D9D9',
                          font=dict(color='#284B63', family="Lucida Console", size=20))

        distplotJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return distplotJSON

    def getLabels(self):
        """
        :return: labels of dataset
        """
        return pd.Series(self.labels).to_json(orient='values')

    def getFeatures(self):
        """
        :return: column names of dataset
        """
        return pd.Series(self.df.columns.values).to_json(orient='values')


# text cleaning
class processText:
    def __init__(self, text, df=None, target=None):
        """
        clean text of single String or entire pandas DataFrame text column.
            Elements to be removed: - URL
                                    - HTML tags
                                    -emojis
                                    - punctuations
                                    - twitter retweet symbols
                                    - expansion of contractions
                                    - remove of stopwords
                                    - remove whitespace or empty text and replace with NaN
                                    - drop NaN values and reset index
        :param text: text column in dataframe or String if only one string need to be cleaned
        :param df: pandas DataFrame
        :param target: target column of dataframe
        """
        if df is not None:
            self.df = df.copy()
        if target is not None:
            self.target = target
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
        return text.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def remove_rt(text):
        return re.sub(r'\brt\b', '', text)

    @staticmethod
    def expand_contractions(text):
        expanded_words = []
        for word in text.split():
            expanded_words.append(contractions.fix(word))
        expanded_text = ' '.join(expanded_words)
        return expanded_text

    @staticmethod
    def stopword(text):
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if word not in stop]
        return " ".join(tokens_without_sw)

    def text_cleaning_df(self):
        self.df[self.text] = [entry.lower() for entry in self.df[self.text]]
        self.df[self.text] = self.df[self.text].apply(lambda x: self.expand_contractions(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_html(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_emoji(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_URL(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_rt(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_punct(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.stopword(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: x.strip())
        self.df.replace(to_replace=r'^s*$', value=np.nan, regex=True, inplace=True)
        self.df.replace(to_replace=r'^$', value=np.nan, regex=True, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.df = self.df.astype(str)

        return self.df

    def text_cleaning_sentence(self):
        self.text = self.remove_html(self.text)
        self.text = self.remove_emoji(self.text)
        self.text = self.remove_URL(self.text)
        self.text = self.remove_rt(self.text)
        self.text = self.remove_punct(self.text)
        self.text = self.stopword(self.text)
        self.text = self.text.lower()
        self.text = word_tokenize(self.text)
        ps = PorterStemmer()
        self.text = [ps.stem(word) for word in self.text]
        wln = WordNetLemmatizer()
        self.text = [' '.join([wln.lemmatize(words) for words in self.text])]

        return self.text


# top bi-grams bar chart
class showBigrams:
    def __init__(self, df, text, target):
        """
        calculate most frequent bi-grams per label of given dataframe and returns the charts as plotly json object
        :param df: pandas DataFrame
        :param text: text column of dataframe
        :param target: target column of dataframe
        """
        self.df = df
        self.target = target
        self.labels = df[self.target].unique()
        self.text = text

    @staticmethod
    def get_top_bi_grams(corpus, n=None):
        vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def plot_bigrams(self):
        color = ['#001219', '#005F73', '#0A9396', '#94D2BD', '#E9D8A6', '#EE9B00', '#CA6702', '#BB3E03', '#AE2012',
                 '#9B2226', '#006466', '#144552', '#212F45', '#312244', '#4D194D']
        bigrams = self.get_top_bi_grams(self.df[self.text], 15)
        y, x = map(list, zip(*bigrams))
        df = pd.DataFrame({'bi-gram': y, 'frequency': x, 'label': 'complete'})

        for label in self.labels:
            bigrams = self.get_top_bi_grams(self.df[self.df[self.target] == label][self.text], 15)
            y, x = map(list, zip(*bigrams))
            d = pd.DataFrame({'bi-gram': y, 'frequency': x, 'label': label})
            df = pd.concat([df, d], ignore_index=True)

        df.sort_values(by='frequency', inplace=True)
        fig = px.bar(df, x='frequency', y='bi-gram', color='label', orientation='h', facet_row='label',
                     color_discrete_sequence=color, text='frequency')
        fig.update_layout(title_text='top bi-grams',
                          showlegend=False,
                          paper_bgcolor='#FFFFFF',
                          plot_bgcolor='#D9D9D9',
                          height=400 * len(df['label'].unique()),
                          font=dict(color='#284B63',
                                    family="Lucida Console", size=15))
        fig.update_yaxes(matches=None)
        fig.update_xaxes(matches=None)

        bigramsJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return bigramsJSON


def lambda_replace(x):
    """
    helper function for bert transformer. Needed to be able to save bert transformer with plotly
    :param x:
    :return:
    """
    return x[0][:, 0, :].squeeze()


class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bert_tokenizer, bert_model, max_length: int = 60,
                 embedding_func: Optional[Callable[[torch.tensor], torch.tensor]] = None):
        """

        :param bert_tokenizer:
        :param bert_model:
        :param max_length:
        :param embedding_func:
        """
        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda_replace

    def _tokenize(self, text: str) -> Tuple[torch.tensor, torch.tensor]:
        # Tokenize the text with the provided tokenizer
        tokenized_text = self.tokenizer.encode_plus(text,
                                                    add_special_tokens=True,
                                                    max_length=self.max_length
                                                    )["input_ids"]
        attention_mask = [1] * len(tokenized_text)

        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str) -> torch.tensor:
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(s) for s in text])

    def fit(self, X, y=None):
        """No fitting necessary"""
        return self


global tokenizer
global bert_model
global bert_transformer


# model training
class trainModels:
    def __init__(self, df, text, target, labels, models, weight_class, average, split, crossValidate, p_label,
                 options=['Tfidf-Vectorizer'], df_test=None):
        """
        :param df: pandas DataFrame
        :param text: text column
        :param target: target column
        :param labels: list of labels to train
        :param models: list of models such as SVM, Logistic Regression, ...
        :param weight_class: boolean, sets sklearn function to weight classes. Used if dataset is unbalanced
        :param average: average method for the metric functions
        :param split: train-test-split distribution
        :param crossValidate: boolean, cross-validate or not
        :param options: list of transformers for the text
        :param df_test: test dataframe
        """
        self.weight_class = weight_class
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.df = df.astype(str)
        self.text = text
        self.target = target
        self.labels = labels
        self.models = models
        self.options = options
        self.average = average
        self.test_split = round(1 - split, 2)
        self.crossValidate = crossValidate
        self.df_test = df_test
        self.p_label = p_label
        self.metrics = {'Accuracy': 'Accuracy', 'Precision': 'Precision', 'Recall': 'Recall', 'F1': 'F1'}
        if len(options) > 2:
            self.metrics.update({'Accuracy': 'A'})
            self.metrics.update({'Precision': 'P'})
            self.metrics.update({'Recall': 'R'})

    @staticmethod
    def style_dataframe(df):
        """
        applies defined styles to given DataFrame.
        :param df: pandas DataFrame
        :return: pandas Style object
        """
        df = df.style.set_table_styles([
            {'selector': 'tr:hover', 'props': [('background-color', '#adb5bd'), ('background', '#adb5bd')]},
            {'selector': 'td:hover', 'props': [('background-color', '#284B63'), ('color', 'white')]},
            {'selector': 'th:not(.index_name)',
             'props': 'background-color: #353535; color: white; text-align: center; font-size: 1.0vw;'},
            {'selector': 'tr', 'props': 'line-height: 4vw'},
            {'selector': 'th.col_heading', 'props': 'text-align: center; font-size: 1.5vw;'},
            {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5vw;'},
            {'selector': 'td', 'props': 'text-align: center; font-weight: bold; color: #353535; font-size: 1vw;'},
            {'selector': 'td,th', 'props': 'line-height: inherit; padding: 0 10px'}],
            overwrite=False) \
            .format('{:.2%}') \
            .apply(lambda x: ["background-color:#3C6E71;" if i == x.max() else "background-color: #D9D9D9;" for i in x],
                   axis=0)

        return df

    def processText(self):
        """
        # further preprocess text -> tokenizer and lemmatizer
        :return: pandas DataFrame
        """
        self.df = self.df[self.df[self.target].isin(self.labels)]
        ps = PorterStemmer()
        self.df[self.text] = [ps.stem(word) for word in self.df[self.text]]
        self.df[self.text] = [word_tokenize(entry) for entry in self.df[self.text]]
        tag_map = defaultdict(lambda: wordnet.NOUN)
        tag_map['J'] = wordnet.ADJ
        tag_map['V'] = wordnet.VERB
        tag_map['R'] = wordnet.ADV

        for i, entry in enumerate(self.df[self.text]):
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                    Final_words.append(word_Final)
            self.df.loc[i, 'clean_text'] = str(Final_words)

        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True, drop=True)

    # define train test split and vectorizer
    def prepareData(self):
        """
        train-test split of data
        fit LabelEncoder
        initialize text transformer
        :return: List of text transformers
        """
        self.processText()
        if not self.df_test:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['clean_text'],
                                                                                    self.df[self.target],
                                                                                    test_size=self.test_split,
                                                                                    random_state=42,
                                                                                    shuffle=True,
                                                                                    stratify=self.df[self.target])
        else:
            self.X_train, self.y_train = self.df[self.text], self.df[self.target]
            self.X_test, self.y_test = self.df_test[self.text], self.df_test[self.target]

        Encoder.fit(self.y_train)
        self.y_train = Encoder.transform(self.y_train)
        self.y_test = Encoder.transform(self.y_test)
        if self.p_label != 1:
            p = np.asarray(self.p_label).reshape(1)
            self.p_label = Encoder.transform(p)[0]
        vectorizer = {}

        if 'Tfidf-Vectorizer' in self.options:
            t = {'Tfidf-Vectorizer': TfidfVectorizer(ngram_range=(2, 2), max_features=10000)}
            vectorizer.update(t)

        if 'N-Gram' in self.options:
            t = {'N-Gram': CountVectorizer(ngram_range=(2, 2), max_features=10000)}
            vectorizer.update(t)

        if 'GloVe' in self.options:
            glove = EmbeddingTransformer('glove')
            t = {'GloVe': glove}
            vectorizer.update(t)

        if 'BERT' in self.options:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            bert_model = BertModel.from_pretrained("bert-base-uncased")
            bert_transformer = BertTransformer(tokenizer, bert_model)
            t = {'BERT': bert_transformer}
            vectorizer.update(t)

        for vec in vectorizer:
            vectorizer[vec].fit(self.df['clean_text'])
        return vectorizer

    # train Models
    def trainPredictionModel(self, model, model_name, vectorizer):
        """
        trains the model with each of the given text transformer and calculate the metrics
        models and vectorizer are used in sklearn pipeline. These are saved with pickle and can be downloaded
        :param model: classifier model
        :param model_name: name of model as string
        :param vectorizer: list of vectorizer
        :return: dict of metrics, list of model and used vectorizer, and if uses cross-validation metrics
        """
        output = {}
        cross_output = {}
        models = []
        if self.average == 'binary':
            ending = ''
        else:
            ending = '_' + self.average
        scoring = ['accuracy', 'precision' + ending, 'recall' + ending, 'f1' + ending]

        for vec in vectorizer.keys():
            if model_name == 'Naive Bayes' and vec in ['GloVe', 'BERT']:
                pipe = Pipeline([
                    (vec, vectorizer[vec]),
                    ('scaler', PredictorScaler),
                    (model_name, model.set_params())
                ])
            else:
                pipe = Pipeline([
                    (vec, vectorizer[vec]),
                    (model_name, model)
                ])

            if self.crossValidate and vec is not 'BERT':
                scores = cross_validate(pipe, self.X_train, self.y_train, scoring=scoring)
                for score in scoring:
                    cross_output.update({(vec, self.metrics[score.partition('_')[0].capitalize()]): mean(scores['test_' + score])})

            pipe.fit(self.X_train, self.y_train)
            pred = pipe.predict(self.X_test)

            file_name = model_name + ' ' + vec + '.pkl'
            with open(os.path.join(app.config['MODEL_FOLDER'], file_name), 'wb') as f:
                pickle.dump(pipe, f)
                models.append(file_name)

            output.update({(vec, self.metrics['Accuracy']): accuracy_score(self.y_test, pred)})
            output.update({(vec, self.metrics['Precision']): precision_score(self.y_test, pred, average=self.average, pos_label=self.p_label)})
            output.update({(vec, self.metrics['Recall']): recall_score(self.y_test, pred, average=self.average, pos_label=self.p_label)})
            output.update({(vec, self.metrics['F1']): f1_score(self.y_test, pred, average=self.average, pos_label=self.p_label)})

        return output, models, cross_output

    # use different models
    def train(self):
        """
        call trainPredictionModel for each selected model and put the model output from these function to one list
        list is used to call the saved pickle objects
        :return: dict of each model with every transformer applied on the text with the metrics
        """
        vectorized = self.prepareData()
        output = {}
        output_cross_val = {}
        models = []

        if self.weight_class:
            comp_class_weight = 'balanced'
        else:
            comp_class_weight = None

        if 'SVM' in self.models:
            output['SVM'], model_name, output_cross_val['SVM'] = self.trainPredictionModel(svm.SVC(C=1.2,
                                                                                                   kernel='linear',
                                                                                                   degree=3,
                                                                                                   gamma='auto',
                                                                                                   class_weight=comp_class_weight),
                                                                                           'SVM',
                                                                                           vectorized)
            models.extend(model_name)

        if 'Naive Bayes' in self.models:
            output['Naive Bayes'], model_name, output_cross_val['Naive Bayes'] = self.trainPredictionModel(
                naive_bayes.MultinomialNB(),
                'Naive Bayes',
                vectorized)
            models.extend(model_name)

        if 'Logistic Regression' in self.models:
            output['Logistic Regression'], model_name, output_cross_val[
                'Logistic Regression'] = self.trainPredictionModel(
                LogisticRegression(class_weight=comp_class_weight),
                'Logistic Regression',
                vectorized)
            models.extend(model_name)

        if 'Random Forest' in self.models:
            output['Random Forest'], model_name, output_cross_val['Random Forest'] = self.trainPredictionModel(
                RandomForestClassifier(n_estimators=100,
                                       random_state=100,
                                       class_weight=comp_class_weight),
                'Random Forest',
                vectorized)
            models.extend(model_name)

        session['trained_models'] = models

        return output, output_cross_val

    # return a table with the calculated metrics
    def plotOutput(self):
        """
        puts the result of train function in pandas DataFrame and applies style
        :return: pandas style object with the calculated metrics for each model and transformer
        """
        out, out_cross_val = self.train()

        df = pd.DataFrame(out).T
        df.sort_index(inplace=True)

        df_style = self.style_dataframe(df)

        if self.crossValidate:
            df_cross_val = pd.DataFrame(out_cross_val).T
            df_cross_val.sort_index(inplace=True)
            df_cross_val_style = self.style_dataframe(df_cross_val)
        else:
            df_cross_val_style = {}

        return df_style, df_cross_val_style


if __name__ == "__main__":
    app.secret_key = secrets.randbelow(999999)
    app.run(debug=True)
