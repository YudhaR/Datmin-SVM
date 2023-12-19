from django.shortcuts import render, reverse, redirect
from django.http import HttpResponseRedirect
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render
from django.contrib import messages
from tablib import Dataset
import csv,io
import re
import numpy as np
import pandas as pd
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary, StopWordRemover
from googletrans.client import Translator


from .models import *
from .forms import *

from django.http.response import JsonResponse

from django.core.serializers import serialize
import json

from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go


def home(request):
    # Fetching data from the Training model
    trainings = Training.objects.all()
    if not trainings:
        return render(request, 'analysis/input_template.html')

    # Creating a DataFrame from the Training model data
    df = pd.DataFrame(list(trainings.values()))

    # Creating the pie chart
    figSentimen = px.pie(df, names='depresi', title='Sentiment', width=400, height=400)
    figSentimen.update_layout(
        paper_bgcolor="#eee",
    )
    chart = figSentimen.to_html()
    words_yes = []
    words_no = []
    kamus = {'the', 'of', 'to', 'is', 'are', 'there', 'a', 'not', 'i'}
    

    for text in df[df['depresi'] == 'Yes']['full_text']:
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            words_yes.extend([word.lower() for word in cleaned_text.split() if word.lower() not in kamus])

    for text in df[df['depresi'] == 'No']['full_text']:
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            words_no.extend([word.lower() for word in cleaned_text.split() if word.lower() not in kamus])
    
    # Counting word occurrences for Yes
    word_count_yes = {}
    for word in words_yes:
        if word in word_count_yes:
            word_count_yes[word] += 1
        else:
            word_count_yes[word] = 1

    # Getting the top 10 words for Yes
    top_words_yes = sorted(word_count_yes.items(), key=lambda x: x[1], reverse=True)[:20]
    df_top_words_yes = pd.DataFrame(top_words_yes, columns=['kata', 'jumlah'])

    # Creating the bar chart for Yes
    fig_word_yes = px.bar(df_top_words_yes, x='kata', y='jumlah', title="Kata Terbanyak Depresi (Yes)")
    fig_word_yes.update_layout(
        xaxis_title="Kata",
        yaxis_title="Jumlah",
    )
    chart_word_yes = fig_word_yes.to_html()

    # Counting word occurrences for No
    word_count_no = {}
    for word in words_no:
        if word in word_count_no:
            word_count_no[word] += 1
        else:
            word_count_no[word] = 1

    # Getting the top 10 words for No
    top_words_no = sorted(word_count_no.items(), key=lambda x: x[1], reverse=True)[:20]
    df_top_words_no = pd.DataFrame(top_words_no, columns=['kata', 'jumlah'])

    # Creating the bar chart for No
    fig_word_no = px.bar(df_top_words_no, x='kata', y='jumlah', title="Kata Terbanyak Depresi (No)")
    fig_word_no.update_layout(
        xaxis_title="Kata",
        yaxis_title="Jumlah",
    )
    chart_word_no = fig_word_no.to_html()

    x = [item.full_text for item in trainings]
    y = [item.depresi for item in trainings]
    sm = SMOTE()

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    svm_classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_classifier.fit(X_train_res, y_train_res)

    y_pred = svm_classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    figConfusionMatrix = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['True Negative', 'True Positive'],
        colorscale='Viridis',
        reversescale=True,
    ))
    

    figConfusionMatrix.update_layout(
        title='Confusion Matrix Heatmap',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
    )

    chartTrain = figConfusionMatrix.to_html()

    target_names = ['no', 'yes']

    classification_rep = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    figClassificationReport = px.bar(
        x=target_names * 3,  # Repeat class names for each metric
        y=[classification_rep[class_name.lower()][metric] for class_name in target_names for metric in ['precision', 'recall', 'f1-score']],
        color=['Precision', 'Recall', 'F1-Score'] * len(target_names),
        barmode='group',
        labels={'y': 'Score', 'x': 'Class', 'color': 'Metric'},
        title='Classification Report Metrics by Class',
    )

    chartClassificationReport = figClassificationReport.to_html()

    visual = {
        'sentimenDataVisual': chart,  
        'wordCountYes': chart_word_yes,
        'wordCountNo': chart_word_no,
        'trainDataVisual' : chartTrain,
        'akurasi' : accuracy_score(y_test, y_pred),
        'reportDataVisual' : chartClassificationReport,
    }


    return render(request, 'analysis/index.html', visual)

def test(request):
    visual = {}
    return render(request, 'analysis/home.html', visual)


def model():
    items = Training.objects.all()

    x = [item.full_text for item in items]
    y = [item.depresi for item in items]
    sm = SMOTE()

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    svm_classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_classifier.fit(X_train_res, y_train_res)
    return svm_classifier, tfidf_vectorizer

def clean(full_text, svm_classifier, tfidf_vectorizer):
    translator = Translator()
    def remove_pattern(text, pattern_regex):
        r = re.findall(pattern_regex, text)
        for i in r:
            text = re.sub(i, '', text)
        return text
            
    clean_tweet = remove_pattern(full_text, " *RT* | *@[\\w]*")
    clean_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\\w+://\\S+)", " ", clean_tweet).split())


    clean_tweet = re.sub(r'\$\w*', '', clean_tweet)
    clean_tweet = re.sub(r'^RT[\s]+', '', clean_tweet)
    clean_tweet = re.sub(r'#', '', clean_tweet)
    clean_tweet = re.sub('[0-9]+', '', clean_tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(clean_tweet)

    tweets_clean = []
    for word in tweet_tokens:
           if (
            word not in stopwords_indonesia and  # remove stopwords
            word not in emoticons and  # remove emoticons
            word not in string.punctuation  # remove punctuation
            ):
            # Apply stemming
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    tweets_clean  = " ".join([char for char in tweets_clean if char not in string.punctuation])

    success_flag = False
    while not success_flag:
        try:
            translated_tweet = translator.translate(tweets_clean, dest='en').text
            success_flag = True  
        except Exception:
            success_flag = False

    x_test = []
    x_test = [translated_tweet] 

    X_tfidf_test = tfidf_vectorizer.transform(x_test)

    y_pred_test = svm_classifier.predict(X_tfidf_test)
    depresi = str(y_pred_test[0])
    return translated_tweet, depresi

def upload_test(request):
    if request.method == 'POST':
        tweet_resource = TweetUploadForm(request.POST, request.FILES)
        new_tweet = request.FILES.get('myfile')
        additional_text = request.POST.get('additional_text')

        svm_classifier, tfidf_vectorizer = model()

        if new_tweet or additional_text:
            if new_tweet:
                data_set = new_tweet.read().decode('UTF-8')
                io_string = io.StringIO(data_set)
                next(io_string)
                for column in csv.reader(io_string):
                    translated_tweet, depresi = clean(column[2], svm_classifier, tfidf_vectorizer)
                    Testing.objects.update_or_create(
                        full_text=column[2],
                        depresi=depresi
                    )

            if additional_text:
                    translated_tweet, depresi = clean(additional_text, svm_classifier, tfidf_vectorizer)
                    Testing.objects.update_or_create(
                        full_text=additional_text,
                        depresi=depresi
                    )
    return redirect('test')



def upload_tweet(request):
    tweets_to_delete = Tweet.objects.all()

    tweets_to_delete.delete()

    cleans_to_delete = Clean.objects.all()

    cleans_to_delete.delete()

    trainings_to_delete = Training.objects.all()
    trainings_to_delete.delete()

    testings_to_delete = Testing.objects.all()
    testings_to_delete.delete()

    if request.method == 'POST':
        tweet_resource = TweetUploadForm()
        dataset = Dataset()
        new_tweet = request.FILES['myfile']

        data_set = new_tweet.read().decode('UTF-8')
        io_string = io.StringIO(data_set)
        next(io_string)
        idt1=1
        for column in csv.reader(io_string):
            created = Tweet.objects.update_or_create(
                    full_text=column[2],
                    username=column[10],
                    idt=idt1,
            )
            idt1+=1
    return redirect('/')

def list_tweet(request):
    items = Tweet.objects.all()
    
    data = {'items': json.loads(serialize('json', items))}
    return JsonResponse(data)

def list_clean(request):
    items = Clean.objects.all()
    
    data = {'items': json.loads(serialize('json', items))}
    return JsonResponse(data)

def list_training(request):
    items = Training.objects.all().order_by('idt')
    
    data = {'items': json.loads(serialize('json', items))}
    return JsonResponse(data)

def list_testing(request):
    items = Testing.objects.all().order_by('-id_twt')
    
    data = {'items': json.loads(serialize('json', items))}
    return JsonResponse(data)

def delete_tweet(request, idt):
    tweets_to_delete = Tweet.objects.filter(idt=idt)

    tweets_to_delete.delete()

    cleans_to_delete = Clean.objects.filter(idt=idt)

    cleans_to_delete.delete()

    trainings_to_delete = Training.objects.filter(idt=idt)

    trainings_to_delete.delete()

    testings_to_delete = Testing.objects.filter(idt=idt)

    testings_to_delete.delete()

    return redirect('/')



#preprocessing data
nltk.download('stopwords')

stopwords_indonesia = stopwords.words('indonesian')

stop_factory = StopWordRemoverFactory().get_stop_words()
more_stopwords = [
    'yg', 'utk', 'cuman', 'deh', 'Btw', 'tapi', 'gua', 'gue', 'lo', 'lu',
    'kalo', 'trs', 'jd', 'nih', 'ntar', 'nya', 'lg', 'gk', 'ecusli', 'dpt',
    'dr', 'kpn', 'kok', 'kyk', 'donk', 'yah', 'u', 'ya', 'ga', 'km', 'eh',
    'sih', 'eh', 'bang', 'br', 'kyk', 'rp', 'jt', 'kan', 'gpp', 'sm', 'usah',
    'mas', 'sob', 'thx', 'ato', 'jg', 'gw', 'wkwk', 'mak', 'haha', 'iy', 'k',
    'tp', 'haha', 'dg', 'dri', 'duh', 'ye', 'wkwkwk', 'syg', 'btw',
    'nerjemahin', 'gaes', 'guys', 'moga', 'kmrn', 'nemu', 'yukkk',
    'wkwkw', 'klas', 'iw', 'ew', 'lho', 'sbnry', 'org', 'gtu', 'bwt',
    'klrga', 'clau', 'lbh', 'cpet', 'ku', 'wke', 'mba', 'mas', 'sdh', 'kmrn',
    'oi', 'spt', 'dlm', 'bs', 'krn', 'jgn', 'sapa', 'spt', 'sh', 'wakakaka',
    'sihhh', 'hehe', 'ih', 'dgn', 'la', 'kl', 'ttg', 'mana', 'kmna', 'kmn',
    'tdk', 'tuh', 'dah', 'kek', 'ko', 'pls', 'bbrp', 'pd', 'mah', 'dhhh',
    'kpd', 'tuh', 'kzl', 'byar', 'si', 'sii', 'cm', 'sy', 'hahahaha', 'weh',
    'dlu', 'tuhh'
]
data = stop_factory + more_stopwords

dictionary = ArrayDictionary(data)
stri = StopWordRemover(dictionary)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)

def prepro(request):

    items = Tweet.objects.all()
    cleans_to_delete = Clean.objects.all()

    cleans_to_delete.delete()

    trainings_to_delete = Training.objects.all()
    trainings_to_delete.delete()

    testings_to_delete = Testing.objects.all()
    testings_to_delete.delete()

    def remove_pattern(text, pattern_regex):
        r = re.findall(pattern_regex, text)
        for i in r:
            text = re.sub(i, '', text)
        return text
    


    cleaned_tweets = []
    translator = Translator()

    for item in items:
        clean_tweet = remove_pattern(item.full_text, " *RT* | *@[\\w]*")
        clean_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\\w+://\\S+)", " ", clean_tweet).split())


        clean_tweet = re.sub(r'\$\w*', '', clean_tweet)
        clean_tweet = re.sub(r'^RT[\s]+', '', clean_tweet)
        clean_tweet = re.sub(r'#', '', clean_tweet)
        clean_tweet = re.sub('[0-9]+', '', clean_tweet)



        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(clean_tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if (
                word not in stopwords_indonesia and  # remove stopwords
                word not in emoticons and  # remove emoticons
                word not in string.punctuation  # remove punctuation
            ):
                # Apply stemming
                stem_word = stemmer.stem(word)
                tweets_clean.append(stem_word)

        # Join the cleaned tokens back into a string
        tweets_clean  = " ".join([char for char in tweets_clean if char not in string.punctuation])
        
        cleaned_tweets.append((item.idt, tweets_clean, item.username))  


    success_flag = False

    for idt, tweet, username in cleaned_tweets:
        while not success_flag:
            try:
                translated_tweet = translator.translate(tweet, dest='en').text
                Clean.objects.create(idt=idt, full_text=translated_tweet, username=username)
                success_flag = True  # Mengatur flag ke True jika iterasi berhasil
            except Exception:
                # Menangkap kesalahan dan melanjutkan ke iterasi berikutnya
                success_flag = False
        success_flag = False
  

    cleaned_tweets = Clean.objects.all()
    return redirect('/')

def label(request):

    items = Clean.objects.all()
    analyser = SentimentIntensityAnalyzer()

    trainings_to_delete = Training.objects.all()
    trainings_to_delete.delete()

    testings_to_delete = Testing.objects.all()
    testings_to_delete.delete()

    cleaned_tweets = []

    for item in items:
        cleaned_tweets.append((item.idt, item.full_text, item.username))

    for idt, tweet, username in cleaned_tweets:
        scores = analyser.polarity_scores(tweet)
        compound_score = scores['compound']

        if compound_score < 0:
            sentiment = 'Negatif'
            depresi = 'Yes'
        elif compound_score == 0:
            sentiment = 'Netral'
            depresi = 'No'
        elif compound_score > 0:
            sentiment = 'Positif'
            depresi = 'No'

        Training.objects.create(idt=idt, full_text=tweet, username=username, compound_score=compound_score, sentiment=sentiment, depresi=depresi)

    return redirect('/')


def latih(request):
    items = Training.objects.all()

    x = [item.full_text for item in items]
    y = [item.depresi for item in items]
    sm = SMOTE()

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    svm_classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_classifier.fit(X_train_res, y_train_res)

    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    new_tweets = ["Finally, I have a little work to do so I can go on holiday"]
    new_tweets_tfidf = tfidf_vectorizer.transform(new_tweets)
    predictions = svm_classifier.predict(new_tweets_tfidf)

    response_data = {
        'predict': predictions.tolist(),
        'accuracy': accuracy,
        'message': 'Training completed successfully',  
    }
    return JsonResponse(response_data, status=200)