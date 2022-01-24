import sys
import nltk
nltk.download(['punkt', 'wordnet'])

# import statements
import re
import pickle
import numpy as np
import pandas as pd
import sqlalchemy as sql


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    Load data from database_filepath
 
    input:
        database_filepath: the database file name and path
    Output:
        X = The messages
        Y = the values of the categories
        col_names = a list with the categories names
    '''

    # load data from database
    engine = sql.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * from `messages`", engine)
    df = df[df.related != 2]

    X = df.message
    Y = df.loc[:,"related":"direct_report"]
    return X, Y, Y.columns


def tokenize(text):
    '''
    Tokenize a message
 
    input:
        text: the message to be tokenized
    Output:
        clean_tokens: a list of tokens from the input
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")

    # lower text
    text = text.lower() 
    
    # remove !letters and !numbers
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build the ML Pipeline
 
    Output:
        pipeline: the pipeline
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model on the test data and print the result
    input:
        model: trained model for evaluation
        X_test: testing features (Unseen data)
        Y_test: true values to compare with prediction on unseen test cases
        category_names: column name of Y_test data
    '''

    Y_test = np.array(Y_test)
    Y_test_predit = model.predict(X_test)

    for idx in range(36):
        print(str(idx + 1) + ": "+ category_names[idx])
        print(classification_report(Y_test[:, idx], Y_test_predit[:, idx]))
        print(" ---------------------------------------------------- ")

def save_model(model, model_filepath):
    ''''
    create the pkl
    input:
        model: the model to be saved
        model_filepath: where to save the file

    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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