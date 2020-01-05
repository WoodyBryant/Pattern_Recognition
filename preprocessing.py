##数据预处理
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split

 ##导入数据，将十个特征分别命名为A到J，所要预测的性别命名为label
df = pd.read_csv('vali_500_with_tag.csv',index_col=False,names
                  = ['A','B','C','D','E','F','G','H','I','J','label'])
#df_ = pd.read_csv('vali_500_with_tag.csv',index_col = False,names = \
#                   ['A','B','C','D','E','F','G','H','I','J','label'])
df__ = pd.read_csv('vali_100_no_tag.csv',index_col=False,names\
                  = ['A','B','C','D','E','F','G','H','I','J'])
df__['label'] =  '$'
df = pd.concat([df,df__])
feature = ['A','B','C','D','E','F','G','H','I','J']
df1 = df.copy()
##将所有特征的值Z-score标准化（均值为0，方差为1）
df1[feature]= pd.DataFrame(preprocessing.scale(df1[feature]))

##保存预测集
df3 = df1[df1['label'] == '$']
df3.to_csv('vali_100_no_tag1.csv',header = 0,index = 0)
df1 = df1[df1['label'] != '$']

#将M，f,F等性别代号转化为数字1和0
df1.label = df1.label.apply(lambda x:1 if x == 'M' else 0)

##根据3σ原则（这里用2.5σ原则）将异常数据替换为均值
df1 = df1.applymap(lambda x: np.nan if abs(x)>3 else x)
mean = df1[feature].apply(lambda x:x.mean())
df1.fillna(mean,inplace = True)

##根据Holdout方法划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(df1[feature], df1['label'], test_size = 0.3,random_state=42)
dataset3_2_train = pd.concat([x_train,y_train], axis=1)
dataset3_2_test = pd.concat([x_test,y_test], axis=1)
df1.to_csv('dataset3_2.csv',header = 0,index = 0)
dataset3_2_train.to_csv('dataset3_2_train.csv',header = 0,index = 0)
dataset3_2_test.to_csv('dataset3_2_test.csv',header = 0,index = 0)
#    
#




    
    





    
