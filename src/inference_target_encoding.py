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
    ## load the model
    for fold in range(5):
        clf = joblib.load(os.path.join(model_path, f"{model_type}_mean_enc_fold{fold}.bin"))
        preds = clf.predict_proba(df.values)[:, 1]
        if fold == 0:
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

predictions= predict('input/test_mean_encoded.csv', model_type= 'xgb', model_path=config.model_output)
sample_sub = pd.read_csv('input/sample_submission.csv')
sample_sub.label = predictions
sample_sub.to_csv('submissions/submssion_xgb_mean_encoded.csv', index=False)