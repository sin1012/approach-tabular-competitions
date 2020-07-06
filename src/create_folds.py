import pandas as pd
import numpy as np
from sklearn import model_selection
import config

df = pd.read_csv('input/train.csv').set_index('id')

def create_folds(df, target):
    ## initialize the kfold column
    df['kfold'] = -1

    ## reset index and randomize
    df.sample(frac=1).reset_index(drop=True)

    ## creating kfold object, let's do 5 fold
    kf = model_selection.StratifiedKFold(n_splits=5)

    ## fill in the kfold column
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=df[target].values)):
        df.loc[valid_idx, 'kfold'] = fold

    df.to_csv('input/train_folds.csv', index=False)

create_folds(df, target=config.target)
