import pandas as pd
import numpy as np
import config
import os
import joblib
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')
def mean_target_encoding(data):
    df = data.copy()

    # num_features
    num_feas = ['age', 'wage per hour', 'capital gains', 'capital losses',
      'dividends from stocks', 'num persons worked for employer',
      'own business or self employed', 'weeks worked in year']

    ## only for this competition, mapping them to 0 and 1 as suggested by the organizer
    ## fill all NaNs
    df[df == ' ?'] = np.nan # the dataset has labelled all NaNs with ? already
    df.y = df.y.map({' - 50000.':0,' 50000+.':1 })

    ## extract all features
    features = [i for i in df.columns if i not in ('kfold', 'y')]

    ## convert all cat cols to string and fill nans with NONE
    for col in features:
        if col not in num_feas:
            df.loc[:,col] = df[col].astype(str).fillna('NONE')
    
    ## label encode the features
    for col in features:
        if col not in num_feas:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])
        
    ## create a list to store 5 validation dataframes
    encoded_dfs = []

    ## go over all folds
    for fold in range(5):
        df_train = df[df.kfold!=fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        for col in features:
            if col not in num_feas:
                mapping_dict = dict(df_train.groupby(col)['y'].mean())
                df_valid.loc[:, col+'_enc'] = df_valid[col].map(mapping_dict)
        ## append to our list of encoded dataframes
        encoded_dfs.append(df_valid)
        ## create full dataframe again and return
    encoded_df = pd.concat(encoded_dfs, axis=0).reset_index(drop=True)  
    return encoded_df

data = pd.read_csv('input/train_folds.csv')
encoded_df = mean_target_encoding(data)
encoded_df.to_csv('input/train_mean_encoded.csv', index=False)