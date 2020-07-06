import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np

### for categorical feature engineering
def feature_engineering(df, cat_feas):
    ## list(itertools.combination([1,2,3],2)) will return [1,2], [1,3], [2,3]
    combination = list(itertools.combinations(cat_feas, 2)) # create pairs of 2
    for c1, c2 in tqdm(combination):
        df[c1+'_'+c2] = df[c1].astype(str) + '_' + df[c2].astype(str)
    print('Feature Engineering Finished.')
    return df

# df = pd.read_csv('input/train_folds.csv')
# df[df == ' ?'] = np.nan
# num_feas = ['age', 'wage per hour', 'capital gains', 'capital losses',
#     'dividends from stocks', 'num persons worked for employer',
#     'own business or self employed', 'weeks worked in year', 'kfold', 'y']
# cat_feas = [i for i in df.columns if i not in num_feas]
# df.y = df.y.map({' - 50000.':0,' 50000+.':1 })
# df_fe = feature_engineering(df, cat_feas)
# df_fe.to_csv('input/train_folds_fe.csv', index=False)