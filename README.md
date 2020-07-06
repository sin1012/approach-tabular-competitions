# <center> Approach Tabular Competitions
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
Private score: `0.90333` (**first place is: `0.90207`**)

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
``
