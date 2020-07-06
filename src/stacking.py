import pandas as pd
import numpy as np
import config
import os
import joblib
from sklearn import metrics
from sklearn import preprocessing
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import time
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

def data_cleaning(train, test):
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

    return train_df, test_df

def stacking(X, y, X_submission):
    # 5折交叉验证
    n_folds = 5
    nseed = 42
    # 正例和负例的比例
    spw = float(len(y)-sum(y))/float(sum(y))

    clfs = [LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, num_leaves=51, min_child_weight=2.5,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective='binary', reg_alpha=1e-05, reg_lambda=0.8,
                scale_pos_weight=spw, metric='auc', boosting='gbdt', seed=nseed, n_jobs=-1),
            LGBMClassifier(n_estimators=100, learning_rate=0.01, max_depth=8, num_leaves=51, min_child_weight=2.5,
                subsample=0.7, subsample_freq=1, colsample_bytree=0.7, objective='binary', reg_alpha=1e-05, reg_lambda=1,
                scale_pos_weight=spw, metric='auc', boosting='gbdt', seed=nseed, n_jobs=-1),
            RandomForestClassifier(n_estimators=500, max_depth=8, bootstrap=True, min_samples_leaf=50, 
                oob_score=True, class_weight='balanced', criterion='gini', random_state=nseed, n_jobs=-1),
            RandomForestClassifier(n_estimators=500, max_depth=8, bootstrap=True, min_samples_leaf=50, 
                oob_score=True, class_weight='balanced', criterion='entropy', random_state=nseed, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=50, class_weight='balanced', 
                criterion='gini', random_state=nseed, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=50, class_weight='balanced', 
                criterion='entropy', random_state=nseed, n_jobs=-1),
            GradientBoostingClassifier(learning_rate=0.01, max_depth=8, max_features='sqrt', n_estimators=500,
                subsample=0.8, min_samples_split=50, min_samples_leaf=10, random_state=nseed),
            XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, min_child_weight=4.5, gamma=0.5,
                subsample=0.6, colsample_bytree=0.6, objective='binary:logistic', reg_alpha=1e-05, reg_lambda=1,
                scale_pos_weight=spw, eval_metric='auc', seed=nseed, n_jobs=-1),
            XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=8, min_child_weight=2.5, gamma=0.1,
                subsample=0.7, colsample_bytree=0.6, objective='binary:logistic', reg_alpha=1e-05, reg_lambda=1,
                scale_pos_weight=spw, eval_metric='auc', seed=nseed, n_jobs=-1)
            ]
    clf_name = ['lgb1','lgb2','rfc1','rfc2','etc1','etc2','gbt','xgb1','xgb2']

    print("Creating train and test sets for stacking.")
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    skf =  StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=nseed)
    for j, clf in enumerate(clfs):
        print('train %d cls: %s' % (j, clf))
        bbtic = time.time()
        dataset_blend_test_j = np.zeros((X_submission.shape[0], n_folds))
        for i, (train, test) in enumerate(skf.split(X, y)):
            print("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            dataset_blend_train[test, j] = clf.predict_proba(X_test)[:, 1]
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        pd.DataFrame(dataset_blend_train, columns=clf_name).to_csv('input/dataset_stack_train.csv', index=None)
        pd.DataFrame(dataset_blend_test, columns=clf_name).to_csv('input/dataset_stack_test.csv', index=None)
        print('%d cls %s cost %ds' % (j, clf.__class__.__name__, time.time() - bbtic))
    
    return None

train = pd.read_csv('input/train.csv').set_index('id')
test = pd.read_csv('input/test_no_label.csv').set_index('id')

train_clean, test_clean = data_cleaning(train, test)

train_clean.to_csv('input/cleaned_train_label_encoded.csv', index=False)
test_clean.to_csv('input/cleaned_test_label_encoded.csv', index=False)

y = train_clean.y.values
X = train_clean.drop('y',axis=1).values
X_submission = test_clean.values

stacking(X,y,X_submission)
