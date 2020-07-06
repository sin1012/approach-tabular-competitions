import pandas as pd
import numpy as np
import config
import os
import joblib
from sklearn import metrics
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')
def mean_target_encoding_test(train, test):
    train_df = train.copy()
    test_df = test.copy()

    # num_features
    num_feas = ['age', 'wage per hour', 'capital gains', 'capital losses',
      'dividends from stocks', 'num persons worked for employer',
      'own business or self employed', 'weeks worked in year']

    ## only for this competition, mapping them to 0 and 1 as suggested by the organizer
    ## fill all NaNs
    train_df[train_df == ' ?'] = np.nan # the dataset has labelled all NaNs with ? already
    test_df[test_df == ' ?'] = np.nan
    train_df.y = train_df.y.map({' - 50000.':0,' 50000+.':1 })

    ## extract all features
    features = [i for i in train_df.columns if i not in ('y','kfold', 'id')]

    ## convert all cat cols to string and fill nans with NONE
    for col in features:
        if col not in num_feas:
            train_df.loc[:,col] = train_df[col].astype(str).fillna('NONE')
            test_df.loc[:,col] = test_df[col].astype(str).fillna('NONE')
    
    ## label encode the features
    for col in features:
        if col not in num_feas:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(train_df[col])
            train_df.loc[:, col] = lbl.transform(train_df[col])
            test_df.loc[:, col] = lbl.transform(test_df[col])

    train_df = train_df.drop('kfold', axis=1)
    test_df = test_df[train_df.drop('y',axis=1).columns]
    for col in features:
        if col not in num_feas:
            mapping_dict = dict(train_df.groupby(col)['y'].mean())
            test_df.loc[:, col+'_enc'] = test_df[col].map(mapping_dict)
    return test_df

train = pd.read_csv('input/train_folds.csv')
test = pd.read_csv('input/test_no_label.csv')
encoded_df = mean_target_encoding_test(train, test)
encoded_df.to_csv('input/test_mean_encoded.csv', index=False)