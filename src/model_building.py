import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Ensure the "logs" directory exists
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if X_train.shape[0]!=y_train.shape[0]:
            raise ValueError('The no of data samples in the X_train and y_train are not same')
        
        logger.debug('initializing Random forest classifier with the model parameters %s',params)
        clf=RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        clf.fit(X_train,y_train)
        logger.debug('model training completed')
        return clf
    
    except ValueError as v:
        logger.error('ValueError during model training: %s', e)
        raise

    except Exception as e:
        logger.error('Model train does not happen due to : %s',e)
        raise

def save_model(model,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('model saved to locaton : %s',file_path)

    except FileNotFoundError as e:
        logger.error('File path not founf:%s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error happened during saving model: %s',e)
        raise

def main():
    try:
        train_data=load_data('./data/processed/train_tfidf.csv')
        X_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values
        params = {'n_estimators':10, 'random_state':42}
        clf=train_model(X_train,y_train,params)
        model_saved_path='models/model.pkl'
        save_model(clf,model_saved_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()





