import sys
import re
from tkinter import N
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import pickle

RS = 42

def load_data(database_filepath):
    '''
    load_data
    Load the data from a database and return data splited into response 
    and explanatory variables.

    Input:  
    database_filepath       filepath to database file

    Returns:
    X, y, category_names
    X       Messages for classifying
    y       Message labels
    category_names      category names corresponding to the message labels
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Disaster_Resp',engine)
    X = df['message'].values
    y = df.drop(['id','message','genre'],axis=1)
    empty_categories = y.columns[y.sum() ==0].values
    y = y.drop(empty_categories,axis=1)
    category_names = y.columns
    y = y.values
    return X, y, category_names

def tokenize(text): 
    '''
    tokenize
    Tokenize the text, removing web addresses, upper case, and anything 
    but letters and digits. 
    Split sentences into list of words.
    Remove stop words and converts verbs into infinitive.
    
    Input:
    text:   message from the data set

    Returns:
    tokens:     list of tokenized words
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, " ", text.lower())
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos = 'v') for word in tokens]
    return tokens


def build_model():
    '''
    build_model
    Build a NLP pipelineclassification model with the following phases:
    - TfidfVectorizer (with the tokenize function)
    - MultiOutputClassifier (with a SGDClassifier)
     
    Input:
    None
    
    Returns:
    NLP pipeline
    '''
    model = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,2))),
        ('sgd2', MultiOutputClassifier(SGDClassifier(class_weight='balanced', random_state=RS, loss='modified_huber'), n_jobs=-1)),
    ])
    return model
  

def save_model(model, model_filepath):
    '''
    save_model
    Save the classification model in a .pkl file

    Input:
    model       The classification to be stored in a file
    model_filepath      the filepath to save the model into
    Returns:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X, Y)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()