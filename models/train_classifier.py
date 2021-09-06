import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer ,TfidfTransformer
import joblib
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def load_data(database_filepath):
    ''''
    Load data from database file path, output feature set, target and target categories
    '''  
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    return  X, Y, Y.columns

def tokenize(text):
    
    clean_tokens=[]
    tokens = word_tokenize(text)
    Lemmatizer=WordNetLemmatizer()
    stop_words= set(stopwords.words("english"))
    for tokn in tokens:
        clean_tok= Lemmatizer.lemmatize(tokn).lower().strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)
    return clean_tokens
 
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''    
    An estimator that can count the text length of each cell in the X
    '''
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        X_len=pd.Series(X).str.len()
        return pd.DataFrame(X_len)

def build_model():
    '''
        Build a pipeline with TFIDF DTM, length of text column, and a random forest classifier.
         Grid search on the `use_idf` from tf_idf and `estimator__n_estimators` from random forest classifier to find the best model
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
         'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]}
    
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performance of each category target column
    '''
    Y_pred=model.predict(X_test)
    Y_pred=pd.DataFrame(Y_pred,columns=category_names)
    for col in category_names:
        print(f'Column Name:{col} \n')
        print(classification_report(Y_test[col], Y_pred[col]))

def save_model(model, model_filepath):
    '''
    Save model to a pickle file
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