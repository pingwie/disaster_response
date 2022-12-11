import json
import plotly
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
def tokenize(text): 
    '''
    tokenize
    Tokenize the text, removing web addresses, upper case, and anything 
    but letters and digits. 
    Split sentences into list of words.
    Remove stop words and converts verbs into infinitive.
    
    Imput:
    text:   message from the data set

    Returns:
    tokens:     list of tokenized words
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, " ", text.lower())
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos = 'v') for word in tokens]
    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster_Resp', engine)
empty_categories = df.drop(['id','message','genre'],axis=1).columns[ df.drop(['id','message','genre'],axis=1).sum() ==0].values
empty_categories
df = df.drop(empty_categories,axis=1)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    genre_counts = df.groupby('genre').count()['message']/df.shape[0]
    genre_names = list(genre_counts.index)
    
    graph_one = []
    df_categ = df.drop(['id','message','genre'],axis=1)
    categ = df_categ.sum().sort_values(ascending=False)
    categ = categ.iloc[:20]/df.shape[0]
    colors = ['lightslategray',] * 20
    colors[0] = 'crimson'
    graph_one.append(
      Bar(
      x = categ.index,
      y = categ.round(2).tolist(),
      marker_color = colors
      )
    )

    layout_one = dict(title = 'Categories in dataset',
                xaxis = dict(title = 'Tags'),
                yaxis = dict(title = 'Pecentage'),
                )
    
    graph_two = []
    colors = ['lightslategray',] * 3
    colors[1] = 'crimson'
    graph_two.append(
      Bar(
      x = genre_names,
      y = genre_counts,
      marker_color = colors
      )
    )

    layout_two = dict(title = 'Source of messages',
                xaxis = dict(title = 'Type',),
                yaxis = dict(title = 'Percentage'),
                width=400,
                height=550
                )
    
    
    
    figures = []

    figures.append(dict(data = graph_one, layout = layout_one))
    figures.append(dict(data = graph_two, layout = layout_two))

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)



    return render_template('master.html',
                           ids=ids,
                           figuresJSON=figuresJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    # vect = TfidfVectorizer(tokenizer=tokenize)
    
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
    app.run()

if __name__ == '__main__':
    main()