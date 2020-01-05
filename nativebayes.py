##最小错误率贝叶斯，但是需要知道先验概率，默认为训练集中某一类的占比，不太准确。
import pandas as pd  
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
##导入测试集，训练集，和所有训练集
df1 = pd.read_csv('dataset3_2_train.csv',index_col=False,names
                  = ['A','B','C','D','E','F','G','H','I','J','label'])
df2 = pd.read_csv('dataset3_2_test.csv',index_col=False,names
                  = ['A','B','C','D','E','F','G','H','I','J','label'])
df3 = pd.read_csv('dataset3_2.csv',index_col=False,names
                  = ['A','B','C','D','E','F','G','H','I','J','label'])
flag = 1
feature = ['A','B','C','D','E','F','G','H','I','J']
feature_ = ['A','E']

    ##版权声明：本段代码为CSDN博主「Scc_hy」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
    ##原文链接：https://blog.csdn.net/Scc_hy/article/details/91869095
def border_of_classifier(sklearn_cl, x, y,n):
        """
        param sklearn_cl : skearn 的分类器
        param x: np.array 
        param y: np.array
        """
        ## 1 生成网格数据
        x_min, y_min = x.min(axis = 0) - 1
        x_max, y_max = x.max(axis = 0) + 1
        # 利用一组网格数据求出方程的值，然后把边界画出来。
        x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01))
        # 计算出分类器对所有数据点的分类结果 生成网格采样
        mesh_output = sklearn_cl.predict(np.c_[x_values.ravel(), y_values.ravel()])
        # 数组维度变形  
        mesh_output = mesh_output.reshape(x_values.shape)
        fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
        ## 会根据 mesh_output结果自动从 cmap 中选择颜色
        plt.subplot(2,1,n)
        plt.pcolormesh(x_values, y_values, mesh_output, cmap = 'rainbow')
        plt.scatter(x[feature_[0]], x[feature_[1]], c = y, s=100, edgecolors ='steelblue' , linewidth = 1, cmap = plt.cm.Spectral)
        plt.xlim(x_values.min(), x_values.max())
        plt.ylim(y_values.min(), y_values.max())
        # 设置x轴和y轴
        plt.xticks((np.arange(np.ceil(min(x[feature_[0]]) - 1), np.ceil(max(x[feature_[0]]) + 1), 1.0)))
        plt.yticks((np.arange(np.ceil(min(x[feature_[1]]) - 1), np.ceil(max(x[feature_[1]]) + 1), 1.0)))
        
        if(n==1):
            plt.title("朴素贝叶斯分类器训练后的分界线(全部样本)")
            plt.savefig("nball")
            
        else:
            plt.title("朴素贝叶斯分类器训练后的分界线(十男十女)")
            plt.savefig("nb10")
           
        plt.show()
        
def modelparaments():
    models = {}
    models['nb'] = GaussianNB()
    features = {}
    features['one'] = feature
    features['two'] = feature_
    df1_F = df1[df1.label == 0]
    df1_F10 = df1_F.sample(n = 10)
    df1_M = df1[df1.label == 1]
    df1_M10 = df1_M.sample(n = 10)
    df1_10 = pd.concat([df1_F10,df1_M10])
    dfs = {}
    dfs['one'] = df1
    dfs['two'] = df1_10
    return models,features,dfs

def drawparaments():
    plt.figure(figsize=(10,40))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
def train(model,df1,df2,feature):
    global flag
    model.fit(df1[feature],df1.label)
    y_pred = model.predict(df2[feature])
    score = (y_pred == df2.label).mean()
    score2 = cross_val_score(model,\
                            df3[feature],df3['label'],cv = 10)\
                            .mean()
    if(flag ==1):
        print("测试集上Holdout验证错误率为（十个特征）{}".format(1-score))
        print("十折交叉验证错误率为（十个特征）{}".format(1-score2))
        
    elif(flag == 2):
        print("训练集上Holdout验证错误率为{}(十个特征，十男十女)".format(1-score))
        
    elif(flag == 3):
        print("测试集上Holdout验证错误率为{},使用的特征为（两个特征）{},{}".format(1-score,feature[0],feature[1]))
        print("十折交叉验证错误率为{},使用的特征为（两个个特征）{},{}".format(1-score2,feature[0],feature[1]))
        
    else:
        print("训练集上Holdout验证错误率为{},使用的特征为（两个特征，十男十女）{},{}".format(1-score,feature[0],feature[1]))
       
        
def main():
    n = 1
    global flag
    drawparaments()
    models,features,dfs = modelparaments()
    for key1 in models:
        for key2 in features:
            for key3 in dfs:
                if(key3 == 'two'):
                    train(models[key1],dfs[key3],dfs[key3],features[key2])
                    flag += 1
                else:
                    train(models[key1],dfs[key3],df2,features[key2])
                    flag += 1
                    
                if(len(features[key2]) == 2):
                    border_of_classifier(models[key1],dfs[key3][features[key2]],\
                                         dfs[key3]['label'],n)
                    n+= 1
                    
main()



        

        
    
 

    
    
