##χ2 （卡方检验)提取特征
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
df1 = pd.read_csv('dataset3_1.csv',index_col=False,names
                  = ['A','B','C','D','E','F','G','H','I','J','label'])
feature = ['A','B','C','D','E','F','G','H','I','J']
df1_new= SelectKBest(chi2, k=2).fit_transform(df1[feature],df1['label'])
print(df1_new)

