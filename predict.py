##svc效果最好
import pandas as pd  
from sklearn.svm import SVC

df1= pd.read_csv('dataset3_2.csv',index_col=False,names
                  = ['A','B','C','D','E','F','G','H','I','J','label'])
df2 = pd.read_csv('vali_100_no_tag.csv',index_col=False,names
                  = ['A','B','C','D','E','F','G','H','I','J'])
df3 = pd.read_csv('vali_100_no_tag1.csv',index_col = False,names = \
                  ['A','B','C','D','E','F','G','H','I','J','label'])


feature = ['A','B','C','D','E','F','G','H','I','J']

def modelparaments():
    model= SVC(kernel = 'rbf')
    features= feature
    df= df1
    
    return model,features,df

def train(model,df1,df3,feature):
    
    model.fit(df1[feature],df1.label)
    y_pred = model.predict(df3[feature])
    df2['label'] = y_pred

def main():
    model,features,df1 = modelparaments()
    train(model,df1,df3,feature)
    
main()
df2.label = df2.label.apply(lambda x:'M' if x == 1 else 'F')
df2.to_csv('vali_100_no_tag.csv',header = 0,index = 0)




        
        
        
    
 

    
    
