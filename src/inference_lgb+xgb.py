import joblib
import pandas as pd 
import numpy as np
import config
import os
import warnings
warnings.filterwarnings('ignore')

def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    predictions = None

    for FOLD in range(5):

        ## read in the test file
        df = pd.read_csv(test_data_path)

        ## load the encoder
        encoders = joblib.load(os.path.join(model_path, f"{model_type}_encoder_fold{FOLD}.pkl"))

        cols = joblib.load(os.path.join(model_path, f"{model_type}_cols_fold{FOLD}.pkl"))
        df[df == ' ?'] = np.nan

        ## encode the testing data
        for c in encoders:
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        df = df[cols]

        ## load the model
        clf1 = joblib.load(os.path.join(model_path, f"{model_type}_fold{FOLD}.bin"))
        clf2 = joblib.load(os.path.join(model_path, f"xgb_fold{FOLD}.bin"))
        preds = clf1.predict_proba(df.values)[:, 1] * .5 + clf2.predict_proba(df.values)[:, 1] * .5

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    ## select decision boundary
    preds = []
    for j in predictions:
        if j < .5:
            preds.append('0')
        else:
            preds.append('1')

    return preds

predictions= predict(config.testing_file, model_type= 'lgb', model_path=config.model_output)
sample_sub = pd.read_csv('input/sample_submission.csv')
sample_sub.label = predictions
sample_sub.to_csv('submissions/submssion_lgb_xgb.csv', index=False)