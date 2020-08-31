#import packages
import pickle
import sys
import joblib
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

import warnings

warnings.simplefilter('ignore')

def load_data(database_filepath):
    '''
    Loads data from SQLite database as a dataframe
    Input:
        database_filepath: File path of database
    Output:
        X: Feature data (messages)
        y: Target variable (categories)
        category_names: List of labels for each category
    '''
    # Load data from database 
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM dmessages', engine)
        

    # Assign feature target variables to X and y
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    # Create category_names list from column headers
    categories = Y.columns.tolist()
    return X, Y, categories
  

def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Input:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed

def build_model():
    
    #builds an ml pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
        
    ])
    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'vect__ngram_range': ((1, 1), (1, 2)),
                  'clf__estimator__bootstrap': (True, False)
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
  
    
def evaluate_metrics(actual, predicted, col_names):
    """Calculate evaluation metrics for ML model
    
    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    col_names: list of strings. List containing names for each of the predicted fields.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i], average = 'weighted')
        recall = recall_score(actual[:, i], predicted[:, i], average = 'weighted')
        f1 = f1_score(actual[:, i], predicted[:, i], average = 'weighted')
        
        metrics.append([accuracy, precision, recall, f1])
        
           
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df

def evaluate_model(model, X_test, Y_test, category_names):
 
    # Predict labels for test dataset
    Y_pred = model.predict(X_test)
    
    # Calculate and print evaluation metrics
    eval_metrics = evaluate_metrics(np.array(Y_test), Y_pred, category_names)
    print(eval_metrics)
   


def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    '''
    joblib.dump(model, model_filepath)



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
   
