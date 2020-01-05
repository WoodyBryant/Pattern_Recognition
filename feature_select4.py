##RandomForestClassifier选取特征
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
df1 = pd.read_csv('dataset3_1.csv',index_col=False,names
                  = ['A','B','C','D','E','F','G','H','I','J','label'])
df1.label = df1.label.apply(lambda x:1 if x == 'M' else 0)
feature = ['A','B','C','D','E','F','G','H','I','J']
forest = RandomForestClassifier(n_estimators=300,
                                random_state=1)
forest.fit(df1[feature],df1['label'])
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(df1[feature].shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature[indices[f]], 
                            importances[indices[f]]))