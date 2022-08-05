# imports
from statistics import mean

from flask import Flask, jsonify, render_template, request, session, send_file
from flask_session import Session
import pandas as pd
import numpy as np
import os
import spacy
import string
import re
import json
import pickle
import shutil
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot

import contractions
import mkl

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer
from sklearn import svm, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from collections import defaultdict

import warnings

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
stop = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_lg")

warnings.simplefilter('ignore')

app = Flask(__name__, template_folder='templates', static_folder='static')
ALLOWED_EXTENSIONS = {'csv'}
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
    session.clear()
    return render_template('index.html')


# upload and safe .csv file in directory
@app.route('/', methods=['POST'])
def upload_file():
    if request.files:
        # upload file
        uploaded_file = request.files['file']
        data_filename = secure_filename(uploaded_file.filename)
        if '.csv' in data_filename:
            uploaded_file = pd.read_csv(uploaded_file, index_col=0)
        if '.tsv' in data_filename:
            uploaded_file = pd.read_csv(uploaded_file, delimiter='\t')
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
    uploaded_df = pd.read_csv(data_file_path, index_col=0)

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
            #df.iloc[:1000].to_csv(data_file_path)
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
    check_box_class_weight = request.form.getlist('class_weight')
    check_box_cross_val = request.form.getlist('cross_val')
    check_box_average = request.form.getlist('average')[0]
    slider_split = int(request.form.getlist('slider')[0]) / 100

    if len(check_box_class_weight) > 0:
        check_box_class_weight = check_box_class_weight[0]
    else:
        check_box_class_weight = 'off'

    if len(check_box_cross_val) > 0:
        check_box_cross_val = True
    else:
        check_box_class_weight = False

    text = session.get('text', None)
    target = session.get('target', None)

    models = trainModels(uploaded_df, text, target, check_box_labels, check_box_models, check_box_class_weight,
                         check_box_average, slider_split, check_box_cross_val, check_box_options)

    model_pred, model_pred_cross_val = models.plotOutput()
    if model_pred_cross_val:
        model_pred_cross_val = model_pred_cross_val.to_html()
    else:
        model_pred_cross_val = ""

    return render_template('view.html', model=model_pred.to_html(),
                           model_pred_cross_val=model_pred_cross_val,
                           labels=check_box_labels)


@app.route('/download')
def download_file():
    path = shutil.make_archive('models', 'zip', os.path.join(app.config['MODEL_FOLDER']))
    return send_file(path, as_attachment=True)


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
        self.df = df
        self.color = ['#284B63', '#4D194D', '#d62828', '#7a0045']
        if target != 0:
            self.target = target
            self.labels = df[target].unique()
        if text != 0:
            self.text = text

    def plotDistribution(self):
        lengths = []
        percentage = []

        for label in self.labels:
            length = self.df[self.df[self.target] == label].shape[0]
            lengths.append(length)
            percentage.append(f'{round((length / self.df.shape[0]) * 100, 2)}%')

        d = {'labels': self.labels, 'samples': lengths, 'percentage': percentage}
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
                name=label,
                marker=dict(color=self.color[np.where(self.labels == label)[0][0]]),
                legendgroup="group"
            ),
                row=1, col=1
            )

            fig.add_trace(go.Histogram(
                x=self.df[self.df[self.target] == label][self.text].str.split().map(lambda x: len(x)),
                name=label,
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

        fig.update_xaxes(range=[0, 200], row=1, col=1)
        fig.update_xaxes(range=[0, 50], row=1, col=2)
        fig.update_xaxes(title_text='length', row=1, col=1)
        fig.update_xaxes(title_text='length', row=1, col=2)
        fig.update_yaxes(title_text='sample number', row=1, col=1)
        fig.update_yaxes(title_text='sample number', row=1, col=2)

        lengthJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return lengthJSON

    def plotWordLength(self):
        hist_data = []
        self.df.dropna(inplace=True)
        for label in self.labels:
            word = self.df[self.df[self.target] == label][self.text].str.split().apply(lambda x: [len(i) for i in x])
            hist_data.append(word.map(lambda x: np.mean(x)))

        fig = create_distplot(hist_data, self.labels, bin_size=0.2, colors=self.color)
        fig.update_xaxes(title_text='length', range=[0, 10])
        # fig.update_yaxes(title_text='probability density')
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
        return pd.Series(self.labels).to_json(orient='values')

    def getFeatures(self):
        return pd.Series(self.df.columns.values).to_json(orient='values')


# text cleaning
class processText:
    def __init__(self, text, df=None, target=None):
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
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_html(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_emoji(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_URL(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_rt(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_punct(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.expand_contractions(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.stopword(x))
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
        wln = WordNetLemmatizer()
        self.text = [' '.join([wln.lemmatize(words) for words in self.text])]

        return self.text


# top bi-grams bar chart
class showBigrams:
    def __init__(self, df, text, target):
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


class W2vVectorizer(object):

    def __init__(self, w2v):
        # Takes in a dictionary of words and vectors as input
        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(glove))])

    # Note: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # it can't be used in a scikit-learn pipeline
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                    or [np.zeros(self.dimensions)], axis=0) for words in X])


# model training
class trainModels:
    def __init__(self, df, text, target, labels, models, weight_class, average, split, crossValidate,
                 options=['Tfidf-Vectorizer']):
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

    # further preprocess text -> tokenizer and lemmatizer
    def processText(self):
        self.df = self.df[self.df[self.target].isin(self.labels) == True]
        self.df.reset_index(inplace=True, drop=True)
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
            self.df.loc[i, 'text_final'] = str(Final_words)

    # train test split and vectorizer defining
    def prepareData(self):
        self.processText()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['text_final'],
                                                                                self.df[self.target],
                                                                                test_size=self.test_split,
                                                                                random_state=42,
                                                                                shuffle=True,
                                                                                stratify=self.df[self.target])
        Encoder.fit(self.y_train)
        self.y_train = Encoder.transform(self.y_train)
        self.y_test = Encoder.transform(self.y_test)
        vectorizer = {}

        if 'Tfidf-Vectorizer' in self.options:
            t = {'Tfidf-Vectorizer': TfidfVectorizer(max_features=5000)}
            vectorizer.update(t)

        if 'N-Gram' in self.options:
            t = {'N-Gram': CountVectorizer(max_features=5000)}
            vectorizer.update(t)

        if 'GloVe' in self.options:
            global glove
            glove = {}
            data = self.df['text_final'].map(word_tokenize).values
            total_vocabulary = set(word for headline in data for word in headline)
            with open('static/files/glove.6B.50d.txt', 'rb') as f:
                for line in f:
                    parts = line.split()
                    word = parts[0].decode('utf-8')
                    if word in total_vocabulary:
                        vector = np.array(parts[1:], dtype=np.float32)
                        glove[word] = vector
            t = {'GloVe': W2vVectorizer(glove)}
            vectorizer.update(t)

        if 'BERT' in self.options:
            return

        for vec in vectorizer:
            if 'GloVe' not in vec:
                vectorizer[vec].fit(self.df['text_final'])
        return vectorizer

    # train Models
    def trainPredictionModel(self, model, model_name, vectorizer):
        output = {}
        cross_output = {}
        models = []
        labels_size = {}
        if self.average == 'binary':
            ending = ''
        else:
            ending = '_' + self.average
        scoring = ['accuracy', 'precision' + ending, 'recall' + ending, 'f1' + ending]

        for label in self.labels:
            labels_size.update({label: self.df[self.df[self.target] == label].shape[0]})

        temp = min(labels_size.values())
        p_label = np.asarray([key for key in labels_size if labels_size[key] == temp]).reshape(1)
        p_label = Encoder.transform(p_label)[0]

        for vec in vectorizer.keys():
            if 'GloVe' in vec:
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

            if self.crossValidate:
                scores = cross_validate(pipe, self.X_train, self.y_train, scoring=scoring)
                for score in scoring:
                    cross_output.update({(vec, score.partition('_')[0].capitalize()): mean(scores['test_' + score])})

            pipe.fit(self.X_train, self.y_train)
            pred = pipe.predict(self.X_test)

            file_name = model_name + ' ' + vec + '.pkl'
            with open(os.path.join(app.config['MODEL_FOLDER'], file_name), 'wb') as f:
                pickle.dump(pipe, f)
                models.append(file_name)

            output.update({(vec, 'Accuracy'): accuracy_score(self.y_test, pred)})

            output.update(
                {(vec, 'Precision'): precision_score(self.y_test, pred, average=self.average, pos_label=p_label)})
            output.update({(vec, 'Recall'): recall_score(self.y_test, pred, average=self.average, pos_label=p_label)})
            output.update({(vec, 'F1'): f1_score(self.y_test, pred, average=self.average, pos_label=p_label)})

        return output, models, cross_output

    # use different models
    def train(self):
        vectorized = self.prepareData()
        output = {}
        output_cross_val = {}
        models = []

        if self.weight_class == 'on':
            comp_class_weight = 'balanced'
        else:
            comp_class_weight = None

        if 'SVM' in self.models:
            output['SVM'], model_name, output_cross_val['SVM'] = self.trainPredictionModel(svm.SVC(C=1.0,
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
            output['Logistic Regression'], model_name, output_cross_val['Logistic Regression'] = self.trainPredictionModel(
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
        out, out_cross_val = self.train()

        df = pd.DataFrame(out).T
        df.sort_index(inplace=True)

        df = df.style.set_table_styles([
            {'selector': 'tr:hover', 'props': [('background-color', '#adb5bd'), ('background', '#adb5bd')]},
            {'selector': 'td:hover', 'props': [('background-color', '#284B63'), ('color', 'white')]},
            {'selector': 'th:not(.index_name)',
             'props': 'background-color: #353535; color: white; text-align: center; font-size: 1.5vw;'},
            {'selector': 'tr', 'props': 'line-height: 4vw'},
            {'selector': 'th.col_heading', 'props': 'text-align: center; font-size: 1.5vw;'},
            {'selector': 'th.col_heading.level0', 'props': 'font-size: 2vw;'},
            {'selector': 'td', 'props': 'text-align: center; font-weight: bold; color: #353535; font-size: 1vw;'},
            {'selector': 'td,th', 'props': 'line-height: inherit; padding: 0 10px'}],
            overwrite=False) \
            .format('{:.2%}') \
            .apply(lambda x: ["background-color:#3C6E71;" if i == x.max() else "background-color: #D9D9D9;" for i in x],
                   axis=0)

        if self.crossValidate:
            df_cross_val = pd.DataFrame(out_cross_val).T
            df_cross_val.sort_index(inplace=True)
            df_cross_val = df_cross_val.style.set_table_styles([
                {'selector': 'tr:hover', 'props': [('background-color', '#adb5bd'), ('background', '#adb5bd')]},
                {'selector': 'td:hover', 'props': [('background-color', '#284B63'), ('color', 'white')]},
                {'selector': 'th:not(.index_name)',
                 'props': 'background-color: #353535; color: white; text-align: center; font-size: 1.5vw;'},
                {'selector': 'tr', 'props': 'line-height: 4vw'},
                {'selector': 'th.col_heading', 'props': 'text-align: center; font-size: 1.5vw;'},
                {'selector': 'th.col_heading.level0', 'props': 'font-size: 2vw;'},
                {'selector': 'td', 'props': 'text-align: center; font-weight: bold; color: #353535; font-size: 1vw;'},
                {'selector': 'td,th', 'props': 'line-height: inherit; padding: 0 10px'}],
                overwrite=False) \
                .format('{:.2%}') \
                .apply(lambda x: ["background-color:#3C6E71;" if i == x.max() else "background-color: #D9D9D9;" for i in x],
                       axis=0)
        else:
            df_cross_val = {}

        return df, df_cross_val


if __name__ == "__main__":
    app.secret_key = "123"
    app.run(debug=True)
