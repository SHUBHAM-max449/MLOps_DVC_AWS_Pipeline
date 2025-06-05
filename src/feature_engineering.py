import pandas as pd
import numpy as np
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml




log_dir='log'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')


file_log_path=os.path.join(log_dir, 'feature_engineering.log')
file_handler=logging.FileHandler(file_log_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
     try:
          with open(params_path,'r') as file:
               params=yaml.safe_load(file)
               logger.debug('parameter files loaded successfully from the path:%s',params_path)
               return params
     except FileNotFoundError as e:
        logger.error("File not found at the path : %s",params_path)
        raise
     except yaml.YAMLError as y:
        logger.error("YAML error: %s",y)
        raise
     except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path:str)-> pd.DataFrame:
     """Load data from a CSV file."""
     try:
          df=pd.read_csv(file_path)
          df.fillna('',inplace=True)
          logger.debug('Data loaded and NaNs filled from %s', file_path)
          return df
     except pd.errors.ParserError as e:
          logger.error('Failed to parse the CSV file: %s', e)
          raise
     except Exception as e:
          logger.error('Unexpected error occurred while loading the data: %s', e)
          raise

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->tuple:
     """Apply TfIdf to the data."""
     try:
          vectorizer=TfidfVectorizer(max_features=max_features)
          X_train=train_data['text'].values
          y_train=train_data['target'].values
          X_test=test_data['text'].values
          y_test=test_data['target'].values

          X_train_bow=vectorizer.fit_transform(X_train)
          X_test_bow=vectorizer.transform(X_test)

          train_df=pd.DataFrame(X_train_bow.toarray())
          train_df['lable']=y_train

          test_df=pd.DataFrame(X_test_bow.toarray())
          test_df['label']=y_test

          logger.debug("tfidf applied and data has been transformed")
          return train_df,test_df
     
     except Exception as e:
          logger.error('Error during BOW transformtion: %s',e)
          raise
     
def save_data(df:pd.DataFrame,file_path:str)->None:
     """Save the dataframe to a CSV file."""
     try:
          os.makedirs(os.path.dirname(file_path),exist_ok=True)
          df.to_csv(file_path,index=False)
          logger.debug("data saved to %s",file_path)
     except Exception as e:
          logger.error('Error while saving the data file :%s',e)
          raise

def main():
     try:
        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data('./data/interim/test_processed.csv')
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features'] 

        train_df,test_df=apply_tfidf(train_data,test_data,max_features)

        save_data(train_df,'./data/processed/train_tfidf.csv')
        save_data(test_df,'./data/processed/test_tfidf.csv')

        logger.debug('Feature engineering completed successfully')

     except Exception as e:
          logger.error('Failed to do feature engineering due to the error : %s',e)
          print(f"Error: {e}")
          raise
     
if __name__=='__main__':
     main()


     
     
     
    
