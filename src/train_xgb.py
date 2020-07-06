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

def train(fold):
   df = pd.read_csv(config.training_file)

   ## label encoding
   ### define numerical columns
   num_feas = ['age', 'wage per hour', 'capital gains', 'capital losses',
      'dividends from stocks', 'num persons worked for employer',
      'own business or self employed', 'weeks worked in year', 'kfold', 'y']

   ## only for this competition, mapping them to 0 and 1 as suggested by the organizer
   df.y = df.y.map({' - 50000.':0,' 50000+.':1 })

   ## define all categorical features
   cat_feas = [i for i in df.columns if i not in num_feas]

   ## fill all NaNs
   df[df == ' ?'] = np.nan # the dataset has labelled all NaNs with ? already
   for col in cat_feas:
      df.loc[:,col] = df[col].astype(str).fillna('NONE') # fill all NaNs wiht None
   
   ## label encoding each column
   ## add each encoder to the dictionary
   encoder = {}
   for col in cat_feas:
      lbl = preprocessing.LabelEncoder()
      lbl.fit(df[col])
      df.loc[:,col] = lbl.transform(df[col])
      encoder[col] = lbl
   
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
   model = xgb.XGBClassifier(n_jobs=-1)

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
      os.path.join(config.model_output, f'xgb_fold{fold}.bin')
   )

   ## save the columns used to fit the model
   joblib.dump(
      df_train.drop(['kfold', 'y'], axis=1).columns,
      os.path.join(config.model_output, f'xgb_cols_fold{fold}.pkl')
   )
   
   ## save the label encoder
   joblib.dump(
      encoder,
      os.path.join(config.model_output, f'xgb_encoder_fold{fold}.pkl')
   )

for i in range(5):
   train(i)


    
    