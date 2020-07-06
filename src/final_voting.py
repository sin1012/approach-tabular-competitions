import glob
import pandas as pd
import numpy as np

preds = []
for file_name in glob.glob('submissions/'+'*.csv'):
    sub = pd.read_csv(file_name)
    preds.append(sub.label)

preds = np.array(preds)
final = []
for i in range(preds.shape[1]):
    votes = preds[:,i]
    vote = max(set(votes), key=list(votes).count)
    if vote == 1:
        final.append('1')
    else:
        final.append('0')


sample_sub = pd.read_csv('input/sample_submission.csv')
sample_sub.label = final
sample_sub.to_csv('submissions/submssion_votes.csv', index=False)