import pandas as pd
import numpy as np

def load_train_data(path):
     try:
        df_train = pd.read_csv(path)
        print("Train data loaded successfully \n ", df_train.head())
        return df_train
     except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None

def load_test_data(path):
      try:
        df_test = pd.read_csv(path)
        return df_test
      except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None    

def dataset_tail(df_train):
    print(df_train.tail())

def data_shape(df_train, df_test):
    print('Training Dataset:',df_train.shape)
    print('Testing Dataset:',df_test.shape)

def cols_name(df_train):
    print(df_train.columns.tolist())

def cols_type(df_train):
    print(df_train.dtypes)

def basic_statistics(df_train):
    print(df_train.describe())

def mis_val_train(df_train):
    print(df_train.isnull().sum().tolist())

def mis_val_test(df_test):
    print(df_test.isnull().sum().tolist())
