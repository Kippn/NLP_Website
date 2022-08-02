# imports

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
import mkl

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import svm, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from collections import defaultdict

from imblearn.combine import SMOTETomek

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
        charts = showCharts(uploaded_df, check_box_text, check_box_target)
        df = processText(check_box_text, uploaded_df).text_cleaning_df()
        df.to_csv(data_file_path)
        #df.iloc[:1000].to_csv(data_file_path)
        out = {}

        if 'Bi-Grams' in check_box_chart:
            if 'bigram' in session:
                bigram = session['bigram']
            else:
                bigram = showBigrams(df, check_box_text, check_box_target).plot_bigrams()
                session['bigram'] = bigram
            out.update({'bigram': bigram})

        if 'Distribution' in check_box_chart:
            if 'dist' in session:
                dist = session['dist']
            else:
                dist = charts.plotDistribution()
                session['dist'] = dist
            out.update({'dist': dist})

        if 'Text Length' in check_box_chart:
            if 'textLength' in session:
                textLength = session['textLength']
            else:
                textLength = charts.plotTextLength()
                session['textLength'] = textLength
            out.update({'textLength': textLength})

        if 'Word Length' in check_box_chart:
            if 'wordLength' in session:
                wordLength = session['wordLength']
            else:
                wordLength = charts.plotWordLength()
                session['wordLength'] = wordLength
            out.update({'wordLength': wordLength})

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
    text = session.get('text', None)
    target = session.get('target', None)

    models = trainModels(uploaded_df, text, target, check_box_labels, check_box_models,
                         check_box_options)
    model_pred = models.plotOutput()

    return render_template('view.html', model=model_pred.to_html(), labels=check_box_labels)


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
        if 'GloVe' not in file_name:
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
            percentage.append(f'{round((length / self.df.shape[0])*100,2)}%')

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
    def __init__(self, text, df=None):
        if df is not None:
            self.df = df.copy()
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
    def stopword(text):
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if word not in stop]
        return " ".join(tokens_without_sw)

    def text_cleaning_df(self):
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_html(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_emoji(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_URL(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_rt(x))
        self.df[self.text] = self.df[self.text].apply(lambda x: self.remove_punct(x))
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


# model training
class trainModels:
    def __init__(self, df, text, target, labels, models, options=['Tfidf-Vectorizer']):
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

    # further preprocess text -> tokenizer and lemmatizer
    def processText(self):
        self.df = self.df[self.df[self.target].isin(self.labels) == True]
        self.df[self.text].dropna(inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.df[self.text] = [entry.lower() for entry in self.df[self.text]]
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
        X, y = [], []
        self.processText()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['text_final'],
                                                                                self.df[self.target],
                                                                                test_size=0.2,
                                                                                random_state=42)
        self.y_train = Encoder.fit_transform(self.y_train)
        self.y_test = Encoder.fit_transform(self.y_test)
        vectorizer = {}

        if 'Tfidf-Vectorizer' in self.options:
            t = {'Tfidf-Vectorizer': TfidfVectorizer(max_features=5000)}
            vectorizer.update(t)

        if 'N-Gram' in self.options:
            t = {'N-Gram': CountVectorizer(max_features=5000)}
            vectorizer.update(t)

        if 'GloVe' in self.options:
            corpus = self.df['text_final'].values
            c_vectorizer = CountVectorizer()
            X = c_vectorizer.fit_transform(corpus)
            CountVectorizedData = pd.DataFrame(X.toarray(), columns=c_vectorizer.get_feature_names())
            CountVectorizedData[self.target] = self.df[self.target]

            # Defining an empty dictionary to store the values
            GloveWordVectors = {}

            # Defining a function which takes text input and returns one vector for each sentence
            def FunctionText2Vec(inpTextData):
                # Converting the text to numeric data
                X = c_vectorizer.transform(inpTextData)
                CountVecData = pd.DataFrame(X.toarray(), columns=c_vectorizer.get_feature_names())

                # Creating empty dataframe to hold sentences
                W2Vec_Data = pd.DataFrame()

                # Looping through each row for the data
                for i in range(CountVecData.shape[0]):
                    # initiating a sentence with all zeros
                    Sentence = np.zeros(50)

                    # Looping through each word in the sentence and if its present in
                    # the Glove model then storing its vector
                    for word in WordsVocab[CountVecData.iloc[i, :] >= 1]:
                        if word in GloveWordVectors.keys():
                            Sentence = Sentence + GloveWordVectors[word]
                    # Appending the sentence to the dataframe
                    W2Vec_Data = W2Vec_Data.append(pd.DataFrame([Sentence]))
                return W2Vec_Data

            # Reading Glove Data
            with open('static/files/glove.6B.50d.txt', 'r', encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.array(values[1:], "float")
                    GloveWordVectors[word] = vector

            WordsVocab = CountVectorizedData.columns
            W2Vec_Data = FunctionText2Vec(self.df['text_final'])

            # Adding the target variable
            W2Vec_Data.reset_index(inplace=True, drop=True)
            W2Vec_Data[self.target] = CountVectorizedData[self.target]

            # Assigning to DataForML variable
            DataForML = W2Vec_Data
            DataForML = DataForML.dropna()

            # Separate Target Variable and Predictor Variables
            TargetVariable = DataForML.columns[-1]
            Predictors = DataForML.columns[:-1]

            X = DataForML[Predictors].values
            y = DataForML[TargetVariable].values

        if 'BERT' in self.options:
            return

        for vec in vectorizer:
            vectorizer[vec].fit(self.df['text_final'])
        return vectorizer, X, y

    # train Models
    def trainPredictionModel(self, model, model_name, vectorizer, X_glove, y_glove):
        output = {}
        models = []
        for vec in vectorizer.keys():
            X_train = vectorizer[vec].transform(self.X_train)
            X_test = vectorizer[vec].transform(self.X_test)
            pipe = Pipeline([
                (vec, vectorizer[vec]),
                (model_name, model)
            ])
            pipe.fit(self.X_train, self.y_train)

            # model.fit(X_train, self.y_train)
            file_name = model_name + ' ' + vec + '.pkl'
            with open(os.path.join(app.config['MODEL_FOLDER'], file_name), 'wb') as f:
                pickle.dump(pipe, f)
                models.append(file_name)

            # model = pickle.load(open(file_name,'rb'))

            # pred = model.predict(X_test)
            pred = pipe.predict(self.X_test)
            output.update({(vec, 'Accuracy'): accuracy_score(self.y_test, pred)})
            if len(self.labels) == 2:
                output.update({(vec, 'Precision'): precision_score(self.y_test, pred, average="binary")})
                output.update({(vec, 'Recall'): recall_score(self.y_test, pred, average="binary")})
                output.update({(vec, 'F1'): f1_score(self.y_test, pred, average="binary")})
            else:
                output.update({(vec, 'Precision'): precision_score(self.y_test, pred, average="macro")})
                output.update({(vec, 'Recall'): recall_score(self.y_test, pred, average="macro")})
                output.update({(vec, 'F1'): f1_score(self.y_test, pred, average="macro")})

        if len(X_glove) != 0:

            # Storing the fit object for later reference
            PredictorScalerFit = PredictorScaler.fit(X_glove)

            # Generating the standardized values of X
            X = PredictorScalerFit.transform(X_glove)

            X_train, X_test, y_train, y_test = train_test_split(X, y_glove, test_size=0.2, random_state=42)

            # Encoder = LabelEncoder()
            y_train = Encoder.fit_transform(y_train)
            y_test = Encoder.fit_transform(y_test)

            pipe = Pipeline([
                (model_name, model)
            ])

            pipe.fit(X_train, y_train)

            file_name = model_name + ' ' + 'GloVe' + '.pkl'
            with open(os.path.join(app.config['MODEL_FOLDER'], file_name), 'wb') as f:
                pickle.dump(pipe, f)
                models.append(file_name)

            pred = pipe.predict(X_test)
            output.update({('GloVe', 'Accuracy'): accuracy_score(y_test, pred)})
            if len(self.labels) == 2:
                output.update({('GloVe', 'Precision'): precision_score(y_test, pred, average="binary")})
                output.update({('GloVe', 'Recall'): recall_score(y_test, pred, average="binary")})
                output.update({('GloVe', 'F1'): f1_score(y_test, pred, average="binary")})
            else:
                output.update({('GloVe', 'Precision'): precision_score(y_test, pred, average="macro")})
                output.update({('GloVe', 'Recall'): recall_score(y_test, pred, average="macro")})
                output.update({('GloVe', 'F1'): f1_score(y_test, pred, average="macro")})

        return output, models

    # use different models
    def train(self):
        vectorized, X_glove, y_glove = self.prepareData()
        output = {}
        models = []
        if 'SVM' in self.models:
            output['SVM'], model_name = self.trainPredictionModel(
                svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), 'SVM',
                vectorized, X_glove, y_glove)
            models.extend(model_name)

        if 'Naive Bayes' in self.models:
            output['Naive Bayes'], model_name = self.trainPredictionModel(naive_bayes.MultinomialNB(), 'Naive Bayes',
                                                                          vectorized,
                                                                          X_glove, y_glove)
            models.extend(model_name)

        if 'Logistic Regression' in self.models:
            output['Logistic Regression'], model_name = self.trainPredictionModel(LogisticRegression(),
                                                                                  'Logistic Regression',
                                                                                  vectorized, X_glove,
                                                                                  y_glove)
            models.extend(model_name)

        if 'Random Forest' in self.models:
            output['Random Forest'], model_name = self.trainPredictionModel(
                RandomForestClassifier(n_estimators=100, random_state=100), 'Random Forest', vectorized, X_glove,
                y_glove)
            models.extend(model_name)

        session['trained_models'] = models

        return output

    # return a table with the calculated metrics
    def plotOutput(self):
        out = self.train()
        df = pd.DataFrame(out).T
        df.sort_index(inplace=True)

        df = df.style.set_table_styles([
            {'selector': 'tr:hover', 'props': [('background-color', '#adb5bd'), ('background', '#adb5bd')]},
            {'selector': 'td:hover', 'props': [('background-color', '#284B63'), ('color', 'white')]},
            {'selector': 'th:not(.index_name)', 'props': 'background-color: #353535; color: white; text-align: center; font-size: 1.5vw'},
            {'selector': 'tr', 'props': 'line-height: 4vw'},
            {'selector': 'th.col_heading', 'props': 'text-align: center; font-size: 2vw'},
            {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5vw;'},
            {'selector': 'td', 'props': 'text-align: center; font-weight: bold; color: #353535; font-size: 1vw'},
            {'selector': 'td,th', 'props': 'line-height: inherit; padding: 0 10px'}],
            overwrite=False) \
            .format('{:.2%}') \
            .apply(lambda x: ["background-color:#3C6E71;" if i == x.max() else "background-color: #D9D9D9;" for i in x],
                   axis=0)

        return df


if __name__ == "__main__":
    app.secret_key = "123"
    app.run(debug=True)
