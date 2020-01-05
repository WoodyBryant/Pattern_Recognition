import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
features = ['age','height','weight','birth_place','live_place'\
            ,'fruit_or_not','main_food','milk1','milk2',\
            'milk3','sex']
all_columns = features.copy()
all_columns.append("label")
df1 = pd.read_csv('data_share.csv',index_col = False,names = all_columns,encoding='gbk')
df1 =df1.dropna()
print("数据集的信息为\n{}".format(df1.info()))
#############数据预处理
df1['birth_place'] = df1['birth_place'].apply(lambda x:x[:2])
df1['live_place'] = df1['live_place'].apply(lambda x:x[:2])
df1['fruit_or_not'] = df1['fruit_or_not'].apply(lambda x:1 if x == 1 or x in 'Yy' else 0)
df1['main_food'] = df1['main_food'].apply(lambda x:1 if x=="米饭" else x)
df1['main_food'] = df1['main_food'].astype("int")
df1['milk1'] = df1['milk1'].apply(lambda x:0 if x in "nN" else 1)
df1['milk2'] = df1['milk2'].apply(lambda x:0 if x in "nN" else 1)
df1['milk3'] = df1['milk3'].apply(lambda x:0 if x in "nN" else 1)
df1['sex'] = df1['sex'].apply(lambda x:0 if x in "Ff" else 1)
df1['height'] = df1['height'].apply(lambda x:x[0]+x[2]+x[3] if '.' in x else x)

df1['height'] = df1['height'].astype("float")
df1['weight'] = df1['weight'].astype("float")
df1 = df1[df1['height']<=200]
df1 = df1[df1['height']>=100]
df1 = df1[df1['weight']<=100]

df1['label'] = df1['label'].astype("int")


#####将省份转换为数字
birth_place = df1['birth_place'].unique()
birth_place = list(birth_place)
v= list(range(38))
birth_place = zip(birth_place,v)
birth_place = dict(birth_place)
print(birth_place)
df1['birth_place'] = df1['birth_place'].apply(lambda x:birth_place[x])

live_place = df1['live_place'].unique()
live_place = list(live_place)
v= list(range(50))
live_place = zip(live_place,v)
live_place = dict(live_place)
df1['live_place'] = df1['live_place'].apply(lambda x:live_place[x])
print(live_place )




print("数据集的信息为\n{}".format(df1.info()))

####独热编码
features_one_hot = ['birth_place','live_place']
for i in features_one_hot:
    dummies = pd.get_dummies(df1[i],prefix = i)
    df1[dummies.columns] = dummies
print("数据集的信息为\n{}".format(df1.info()))
features = df1.columns
features = list(features)

####移除标签栏
features.remove("fruit_or_not")

features.remove("birth_place")
features.remove("live_place")
#model = lgb.LGBMClassifier(num_class = 3)
model = lgb.LGBMRegressor()
x_train, x_test, y_train, y_test = train_test_split(df1[features],df1['fruit_or_not'], test_size = 0.3,random_state=44)
model.fit(x_train,y_train)
y_pred1 = model.predict(x_test)
#score = (y_pred1 == y_test).mean()
score = roc_auc_score(y_test,y_pred1)
y_test  = list(y_test)
y_pred1  = list(y_pred1)

print("{}测试集上auc为:{}".format(model,score))
df1.to_csv("preprocess.csv",index = False)
