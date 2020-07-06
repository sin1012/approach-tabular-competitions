import pandas as pd
import numpy as np
import config
import os
import joblib
from sklearn import metrics
from sklearn import preprocessing
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


def train(fold):
   df = pd.read_csv('input/train_mean_encoded.csv')
   
   ## create data for training and validation
   df_train = df[df.kfold != fold].reset_index(drop=True)
   df_valid = df[df.kfold == fold].reset_index(drop=True)

   ## prepare data for training
   x_train = df_train.drop(['kfold', 'y'], axis=1).values
   y_train = df_train[config.target].values

   ## similarly, we prepare data for testing
   x_valid = df_valid.drop(['kfold', 'y'], axis=1).values
   y_valid = df_valid[config.target].values

   ## initialize a model
   model = lgb.LGBMClassifier(n_jobs=-1, n_estimators=200, learning_rate=.05, subsample_freq=1, subsample=.8)

   ## fit
   model.fit(x_train, y_train)

   ## predict on validation dataset
   valid_preds = model.predict_proba(x_valid)[:,1]

   ## get roc auc score
   auc = metrics.roc_auc_score(y_valid, valid_preds)

   ## print auc  
   print(f"Fold = {fold}, AUC = {auc}") 

   ## save the model
   joblib.dump(
      model,
      os.path.join(config.model_output, f'lgb_mean_enc_fold{fold}.bin')
   )

   ## save the columns used to fit the model
   joblib.dump(
      df_train.drop(['kfold', 'y'], axis=1).columns,
      os.path.join(config.model_output, f'lgb_mean_enc_cols_fold{fold}.pkl')
   )

for i in range(5):
   train(i)


    
    