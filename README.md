# <center> Approach Tabular Competitions

## Summary
This is a kaggle in class competition using the census-income dataset(42 columns). The link is: https://www.kaggle.com/c/ml2020spring-hw2. There are many categorical variables in this dataset, so I think it's a good way to demonstrate how to approach tabular competitions. My submissions aren't made during the competition but I was able to beat the first place(`0.90217`) in this competition using a single **XGBoost** model(`0.90225`). Later on, I demonstrated different techniques like **feature engineering**, **target encoding** and **stacking** to achieve even higher score(`0.90383`). Overall, I didn't really spend much effort but enough to illustrate the approaches to tabular competitions.   
  
## Training log
In this section, we will cover the past experiments.
- **XGB** with categorical features only + default parameters:
```
Fold = 0, AUC = 0.9017815887538851
Fold = 1, AUC = 0.9012713246359005
Fold = 2, AUC = 0.9029613005680674
Fold = 3, AUC = 0.900086372675337
Fold = 4, AUC = 0.8939898298190413
```
Private score: `0.86974`
  
Public score: `0.87227`
  
- **XGB** with all features + default parameters:
```
Fold = 0, AUC = 0.9359181303448872
Fold = 1, AUC = 0.936000110274057
Fold = 2, AUC = 0.9351370597295269
Fold = 3, AUC = 0.9328126958729935
Fold = 4, AUC = 0.9306190223788716
```
Private score: `0.90225`

Public score: `0.90464`

- **LGB** with all features + default parameters:
```
Fold = 0, AUC = 0.936719286622042
Fold = 1, AUC = 0.936488801201363
Fold = 2, AUC = 0.9358271568591243
Fold = 3, AUC = 0.9331580305261477
Fold = 4, AUC = 0.9308395965009834
```
Private score: `0.90145`

Public score: `0.90145`

- **LGB** with all features + target encoding + default parameters:
```
Fold = 0, AUC = 0.9374316652794135
Fold = 1, AUC = 0.937860646882183
Fold = 2, AUC = 0.9362173553680317
Fold = 3, AUC = 0.9322454346800466
Fold = 4, AUC = 0.9306594908771625
```
Private score: `0.90333` 

Public score: `0.90495`

- **LGB** with all features + target encoding +  n_estimators=200, learning_rate=.05, subsample_freq=1, subsample=.7:
```
Fold = 0, AUC = 0.9381164617831714
Fold = 1, AUC = 0.9374116442121987
Fold = 2, AUC = 0.9369765558395055
Fold = 3, AUC = 0.933722170755216
Fold = 4, AUC = 0.931267766737079
```
Private score: `0.90303`

Public score: `0.90483`

- **Stacking** 

First level:
```python
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
```
Second level:
```python
model = LGBMRegressor(max_depth=3)
```
Private score: `0.90383`(**first place is: `0.90207`**)

Public score: `0.90337`

## Deal with Categorical features
- Label Encoding
```python
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
```
- One-hot Encoding
can be easily done using `get_dummy()`, however it's not good in this competition as we have many categorical variables.

## Feature Engineering
There's no good rule-of-thumb for feature of engineering as it all comes down to your **understanding of the data** and your **domain knowledge**. Hence, I can only introduce some possible ways that usually works. 

- For categorical variable:
```python
def feature_engineering(df, cat_feas):
    ## list(itertools.combination([1,2,3],2)) will return [1,2], [1,3], [2,3]
    combination = list(itertools.combinations(cat_feas, 2)) # create pairs of 2
    for c1, c2 in tqdm(combination):
        df[c1+'_'+c2] = df[c1].astype(str) + '_' + df[c2].astype(str)
    print('Feature Engineering Finished.')
    return df
```

- For numerical variable:

## Target Encoding
Target encoding is a technique in which you map each category in a given  feature to its mean target value, but this must always be done in a **cross-validated**  manner. It means that the first thing you do is create the folds, and then use those  folds to create target encoding features for different columns of the data in the same  way you fit and predict the model on folds. Here are what I have used.
```python
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
```

## Stacking
If you wanna get even better, stack!! Stack multiple diverse models and create a second level model(usually not a complicated model).
```python
def stacking(X, y, X_submission):
    n_folds = 5
    nseed = 42
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
```
