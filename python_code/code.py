
# coding: utf-8

# 
# 

#                                   Data Collection
# 
# Download audio recording from the following website: https://web.archive.org/web/20150417081759/http://www.911dispatch.com/tape-library/
# 
#  
# 

# In[ ]:


cd /home/hduser/audio_data/High


# Functions to transcribe audio file

# In[ ]:


import speech_recognition as sr # import speech recognition module 
from os import path

# Function to transcribe audio using IBM speech-to-text API.
def IBM(x):
    audio_file = path.join(path.dirname(path.realpath(x)), x)
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    IBM_USERNAME = "ba305638-8b4b-479c-b6f9-6c3db42261e6"  
    IBM_PASSWORD = "qicSAYXGLKLy"
    text = r.recognize_ibm(audio,username=IBM_USERNAME,password=IBM_PASSWORD)
    file = open("High_good_ibm.txt", "a")
    y = str(x)
    file.write(y + "\n" + str(text) + "\n")
    file.close()
    print('File Saved')
    return 

'''Initialize google cloude with 'gcloud init' then login into the account '''

# Function to transcribe using Google speech-to-text API.
def google(x):
    audio_file = path.join(path.dirname(path.realpath(x)), x)
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = None
    text = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
    file = open("High_Good_google.txt", "a")
    y = str(x)
    file.write(y + "\n" + str(text) + "\n")
    file.close()
    print('File Saved')
    return 

# Function to transcribe using CMU ASR.
def cmu(x):
    audio_file = path.join(path.dirname(path.realpath(x)), x)
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    text = r.recognize_sphinx(audio)
    file = open("High_Good_cmu.txt", "a")
    y = str(x)
    file.write(y + "\n" + str(text) + "\n")
    file.close()
    print('File Saved')
    return


# In[ ]:


# Call functions to transcribe audio
IBM('1.flac')
google('1.flac')
cmu('1.flac')
# File is saved in txt format in the working directory.


# Create and a csv file format with attribute name as IncidentID, Text and Class for each ASR output. 
# 
# Preprocess the dataset
# 
# Pre-processing step
# 1. Regular expression
# 2. lower case
# 3. stemming
# 4. Remove stopwords
# 
# Functions to pre-process text
# 

# In[ ]:


# Import nltk modules for pre-processing

import nltk # Import Natural Language Toolkit module
import re # Import regular expression library
from nltk.corpus import stopwords # import stopwords

# create a set of pre-defined stopwords from nltk for speed 
stopwords = set(nltk.corpus.stopwords.words('english'))

# Update stopwords list
stopwords.update(['911','like', 'name', 'okay',
                'ok', 'coming', 'could', 'days', 'everyone',
                'get', 'give', 'going', 'liked', 'say', 'th',
                'still', 'vs','call','operator',
                'phone','hello','nine', 'address','one','building'])


# create a stemmer variable. Assign this variable snowballstemmer.
stemmer = nltk.stem.SnowballStemmer('english')

# Set regular expression pattern to pattern variable
pattern = r"(?u)\b\w\w+\b"

# Function for stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

# Wrapped function for tokenization, Regular_expression, Stop words removal and stemming 
def preprocess_transcripts(transcript,
                    token_pattern = pattern,
                    exclude_stopword=True,
                    stem=True):
    # stop words are not removed for word2vec average features vectors
    token_pattern = re.compile(token_pattern, flags = 0 )
    tokens = [x.lower() for x in token_pattern.findall(transcript)]
    tokens_stemmed = tokens
    if stem:
        tokens_stemmed = stem_tokens(tokens, stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]

    return tokens_stemmed


# Class for feature extraction

# In[ ]:


import numpy as np # import modules
import random

class features(object):
    
    # Bag of words feature vectors. Binary occurence marker of features are used.
    def bow(train, test, num_features):
        from sklearn.feature_extraction.text import CountVectorizer   
        # import countvectorizer module
        vectorizer = CountVectorizer(analyzer = "word",
                                     tokenizer = None,
                                     binary = True,
                                     preprocessor = None,
                                     stop_words = None,
                                     max_features = num_features)
        # fit the training dataset
        train_bw = vectorizer.fit_transform(train.map(lambda x: ' '.join(x)))
        test_bw = vectorizer.transform(test.map(lambda x: ' '.join(x)))
        print('Vectorized')
        return train_bw, test_bw # return bag-of-words feature vectors
    
    # Tf-idf vectorization function
    def tfidf(train, test, num_features):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Create unigrams
        vectorizer = TfidfVectorizer(sublinear_tf=True,  
                                 max_df=0.8, 
                                 min_df=2,
                                 ngram_range=(1,2), 
                                 norm = 'l1',
                                 max_features = num_features)
        train_tf = vectorizer.fit_transform(train.map(lambda x: ' '.join(x)))
        test_tf = vectorizer.transform(test.map(lambda x: ' '.join(x)))
        print('Vectorized')
        return train_tf, test_tf # return tfidf features
    
    
    '''Class to create word2vec average vectors'''
class word2vec(object):
    # Function to average all of the word vectors in a given transcript
    def AvgFeatures(words, model, num_features):
        # initialize an empty numpy array
        feature = np.zeros((num_features,),dtype="float32")
        xwords = 0.
        
        #index2word is a list od words in the model's vocabulary. convert it into set.
        model_words_set = set(model.wv.index2word)
        
        # Loop over each word in the transcript and if that word is in the model's vocabulory, 
        # add its feature vector to the total in numpy array 
        for word in words:
            if word in model_words_set:
                xwords = xwords + 1.
                feature = np.add(feature, model[word])
        # Divide the features by the total number of words in the transcript.
        avgfeatures = np.divide(feature, xwords)
        return avgfeatures

    # load pre-trained model
    def load_model(trained_model):
        from gensim.models import Word2Vec
        model = gensim.models.KeyedVectors.load_word2vec_format(trained_model, binary=True)
        print('Model Loaded')
        return model             


# Change directory to the dataset directory
# 
# Import train and test dataset 

# In[ ]:


cd /home/hduser/dataset


# In[ ]:


# Function for pandas dataframe
def pandas_DataFrame(filename):
    import pandas as pd
    return pd.DataFrame.from_csv(filename, sep=',' , encoding='utf-8')

# import train dataset
ibm = pandas_DataFrame('ibm.csv')
cmu = pandas_DataFrame('cmu.csv')
google = pandas_DataFrame('google.csv')

# import test dataset
test = pandas_DataFrame('google.csv')

# print first five rows
test.head()


# In[ ]:


# Randomize dataset
ibm = ibm.iloc[np.random.permutation(len(ibm))]
cmu = cmu.iloc[np.random.permutation(len(cmu))]
google = google.iloc[np.random.permutation(len(google))]

test = test.iloc[np.random.permutation(len(test))]
# print first five rows
test.head()


#                             Create unigrams from train and test dataset
# Get list of unigrams from train and test dataset using map function.
# 

# In[ ]:


# Unigrams of training dataset
# IBM
ibm['unigram'] = ibm['Text'].map(lambda x: preprocess_transcripts(x))

# Google
google['unigram'] = google['Text'].map(lambda x: preprocess_transcripts(x))

# cmu
cmu['unigram'] = cmu['Text'].map(lambda x: preprocess_transcripts(x))

# unigrams of test dataset
test['unigram'] = test['Text'].map(lambda x: preprocess_transcripts(x))

# print unigrams of test dataset
test.head(5)


#                         Vectorize using Bag-of-words
# Create sparse matrix of feature vectors for train and test dataset
# 

# In[ ]:


# Creating feature vectors for training and test dataset
X_train_ibm, X_test = features.bow(ibm['unigram'] , test['unigram'], 1000) # using 1000 feature i.e. vocabulary list
X_train_google, X_test = features.bow(google['unigram'], test['unigram'], 1000)
X_train_cmu, X_test = features.bow(cmu['unigram'], test['unigram'], 1000)


#                         Vectorize using tf-idf vectors
# Create spare matrix of tf-idf feature vectors for Google dataset                        

# In[ ]:


# Using 1000 features
x_train_google_tf, X_test_tf = features.tfidf(google['unigram'], test['unigram'], 1000)


#                         Take average word2vec features
# Create average feature vectors for Google transcript                        
#                         

# In[ ]:


# Load pre-trained model
# Loading takes about 10 mins depending upon the memory avaiable. 
# Download pre-trained word2vec model using: 'git clone https://github.com/mmihaltz/word2vec-GoogleNews-vectors' 
# in terminal
import gensim
model = word2vec.load_model('/home/hduser/model/GoogleNews-vectors-negative300.bin')
num_features = 300 # take 300 features


# In[ ]:


# take average vectors
X_train_google_wv = word2vec.AvgFeatures(google, model, num_features )
X_test_wv = word2vec.AvgFeatures(test, model, num_features )

from sklearn.preprocessing import normalize
# Normalise
X_train_google_wv =normalize(X_train_google_wv)
X_test_wv = normalize(X_test_wv)


# In[ ]:


# Create targets
y_train_ibm = ibm['Class'].values
y_train_google = google['Class'].values
y_train_cmu = cmu['Class'].values
y_test = test['Class'].values


#                                     benchmark classifiers

# In[ ]:


# Import Sklearn classifier module
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Import sklearn metrics4
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

import sys
# Use for Graphs
import matplotlib.pyplot as plt


labels = ['High', 'Medium', 'Low']

def compare(clf):
    print('=' * 80)
    print('Training on IBM dataset')
    clf.fit(X_train_ibm, y_train_ibm) # Fit the data for training
    pred_ibm = clf.predict(X_test)
    acc_ibm = metrics.accuracy_score(y_test, pred_ibm) # Get accuracy
    f_ibm = f1_score(y_test, pred_ibm, average='weighted') # Get f-score
    pre_ibm = precision_score(y_test, pred_ibm, average='weighted') # Get Precision score
    re_ibm = recall_score(y_test, pred_ibm, average='weighted')  # Get Recall score
    print('Accuracy on IBM:', acc_ibm)
    print('F-score on IBM:', f_ibm)
    print('Precision on IBM:', pre_ibm )
    print('Recall on IBM:', re_ibm )
    
    print('-' * 80)
    print('Training on Google dataset')
    clf.fit(X_train_google, y_train_google) # Fit the data for training
    pred_google = clf.predict(X_test)
    acc_google = metrics.accuracy_score(y_test, pred_google) # Get accuracy
    f_google = f1_score(y_test, pred_google, average='weighted')
    pre_google = precision_score(y_test, pred_google, average='weighted')
    re_google = recall_score(y_test, pred_google, average='weighted')  
    print('Accuracy on Google:', acc_google)
    print('F-score on Google:', f_google)
    print('Precision on Google:', pre_google )
    print('Recall on Google:', re_google )
    
    print('-' * 80)
    print('Training on CMU dataset')
    clf.fit(X_train_cmu, y_train_cmu) # Fit the data for training
    pred_cmu = clf.predict(X_test)
    acc_cmu = metrics.accuracy_score(y_test, pred_cmu) # Get accuracy
    f_cmu = f1_score(y_test, pred_cmu, average='weighted')
    pre_cmu = precision_score(y_test, pred_cmu, average='weighted')
    re_cmu = recall_score(y_test, pred_cmu, average='weighted')  
    print('Accuracy on CMU:', acc_cmu)
    print('F-score on CMU:', f_cmu)
    print('Precision on CMU:', pre_cmu)
    print('Recall on CMU:', re_cmu )
    
    print('-' * 80)
    print('Training on tfidf vectors')
    clf.fit(x_train_google_tf, y_train_google) # Fit the data for training
    pred_google_tf = clf.predict(X_test_tf)
    acc_google_tf = metrics.accuracy_score(y_test, pred_google_tf) # Get accuracy
    f_google_tf = f1_score(y_test, pred_google_tf, average='weighted')
    pre_google_tf = precision_score(y_test, pred_google_tf, average='weighted')
    re_google_tf = recall_score(y_test, pred_google_tf, average='weighted')  
    print('Accuracy on Google:', acc_google_tf)
    print('F-score on Google:', f_google_tf)
    print('Precision on Google:', pre_google_tf )
    print('Recall on Google:', re_google_tf )
    
    
    print('-' * 80)
    print('Training on word2vec avg vectors')
    clf.fit(X_train_google_wv, y_train_google) 
    pred_google_wv = clf.predict(X_test_wv)
    acc_google_wv = metrics.accuracy_score(y_test, pred_google_wv) 
    f_google_wv = f1_score(y_test, pred_google_wv, average='weighted')
    pre_google_wv = precision_score(y_test, pred_google_wv, average='weighted')
    re_google_wv = recall_score(y_test, pred_google_wv, average='weighted')  
    print('Accuracy on Google:', acc_google_wv)
    print('F-socre on Google:', f_google_wv)
    print('Precision on Google:', pre_google_wv)
    print('Recall on Google:', re_google_wv)
    print('-' * 80)
    clf_descr = str(clf).split('(')[0]
    cls = clf_descr

    return clf_descr, ibm_acc, google_acc, cmu_acc, acc_google_tf, acc_google_wv


#                             Train and test classifiers 

# In[ ]:


# Train and test classifiers 

# Results from SVM
results = []
print('=' * 80)
print("SVM")
results.append(compare(LinearSVC(penalty='l1', dual=False, 
                                   multi_class='ovr',
                                   tol=1e-3)))
# Results from AdaBoost
print('=' * 80)
print('AdaBoost')
results.append(compare(AdaBoostClassifier(n_estimators=256, algorithm='SAMME.R')))

# Results from Bernoulli Naive Bayes
print('=' * 80)
print("BernoulliNB")
results.append(compare(BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)))
# Results from Logistic regression
print('=' * 80)
print('Logistic Regression')
results.append(compare(LogisticRegression(solver='liblinear', 
                                            penalty='l1', 
                                            class_weight='balanced',
                                            multi_class = 'ovr')))


#                                         Print Graphs

# In[ ]:


# Print Graph of comparison of accuracy of different Feature Vectors from Google
indices = np.arange(len(results))
results = [[x[i] for x in results] for i in range(4)]
clf_names, ibm_acc, google_acc, cmu_acc = results
plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, ibm_acc, .2, label="Bag Of Words", color='maroon')
plt.barh(indices + .3, google_acc, .2, label="tf-idf",
         color='green')
plt.barh(indices + .6, cmu_acc, .2, label="word2vec", color='blue')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)
plt.xlabel('Accuracy')

for i, c in zip(indices, clf_names):
    plt.text(-.2, i, c)
plt.show()


# Print Graph for Accuracy on ASR datasets
indices = np.arange(len(results))
results = [[x[i] for x in results] for i in range(4)]
clf_names, ibm_acc, google_acc, cmu_acc = results
plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, ibm_acc, .2, label="Google", color='maroon')
plt.barh(indices + .3, google_acc, .2, label="IBM",
         color='green')
plt.barh(indices + .6, cmu_acc, .2, label="CMU", color='blue')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)
plt.xlabel('Accuracy')

for i, c in zip(indices, clf_names):
    plt.text(-.2, i, c)
plt.show()

