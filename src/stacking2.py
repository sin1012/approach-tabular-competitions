import pandas as pd
import numpy as np
import config
import os
import joblib
from sklearn import metrics
from sklearn import preprocessing
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import time
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

label = pd.read_csv('input/cleaned_train_label_encoded.csv').y
train_stack_lv1 = pd.read_csv('input/dataset_stack_train.csv')
test_stack_lv1 = pd.read_csv('input/dataset_stack_test.csv')
model = LGBMRegressor(max_depth=3)
model.fit(train_stack_lv1, label)
predictions = model.predict(test_stack_lv1)
print(predictions[:100])
preds = []
for j in predictions:
    if j < .5:
        preds.append('0')
    else:
        preds.append('1')

sample_sub = pd.read_csv('input/sample_submission.csv')
sample_sub.label = preds
sample_sub.to_csv('submissions/submssion_stacking.csv', index=False)
