##RFE
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd
df1 = pd.read_csv('dataset3_1.csv',index_col=False,names
                  = ['A','B','C','D','E','F','G','H','I','J','label'])
df1.label = df1.label.apply(lambda x:1 if x == 'M' else 0)
feature = ['A','B','C','D','E','F','G','H','I','J']
estimator = SVR(kernel="linear")
selector = RFE(estimator,2, step=1)
selector = selector.fit(df1[feature],df1['label'])
print(selector.support_)

