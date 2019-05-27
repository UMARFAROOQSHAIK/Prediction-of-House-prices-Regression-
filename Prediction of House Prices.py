#!/usr/bin/env python
# coding: utf-8

# ### EXPLORATORY DATA ANALYSIS

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\Password\Desktop\Pucho technologies\train.csv")


# In[3]:


pd.set_option('display.max_columns',100)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


#Replacing the null values with no since null valleys are Alleys having no alleys
df['Alley']=df['Alley'].fillna('No')


# In[7]:


df.describe()


# In[8]:


df.describe(include=object)


# In[ ]:





# In[ ]:





# ### VISUALISING ALL THE OBJECT TYPES AND THEIR PERCENTAGES

# In[ ]:





# In[9]:


for x in df.select_dtypes(include=object):
    plt.figure(figsize=(6,5))
    sns.countplot(y=x,data=df)
    plt.show()
    a=[]
    b=[]
    m=0
    for y in df[x].unique():
        size=len(df[x].unique())
        m+=1
        a.append(len(df.loc[df[f'{x}']==f'{y}',f'{x}']))
        if m==size:
            print('The percentage of entities')
            for z in a:
                print(z/(sum(a))*100)
            break
            


# In[10]:


for x in df.select_dtypes(exclude=object):
    sns.boxplot(y=x,data=df)
    plt.show()

LOT AREA,MASVNAREA,BSMTFINSF2,LOWQUALFINSF,GARAGEAREA,WOODDECKSF,
OPENPRCHSF,ENCLOSEDPORCH,3SSNPORCH,SCREENPORCH,POOLAREA,MISCVAL,SALEPRICE 
has large no of outliers
# ## Data cleaning

# In[11]:


df=df.drop_duplicates()    ## dropping the duplicates


# In[12]:


df.shape


# In[13]:


df['LotFrontage'].sort_values(ascending=False).head(5)  ## checking outliers


# In[13]:


df=df[df.LotFrontage<313]


# In[ ]:


df['LotFrontage'].sort_values(ascending=False).head()


# In[ ]:


df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[ ]:


df['MasVnrArea'].sort_values(ascending=False).head()


# In[ ]:


df=df[df.MasVnrArea<1378]


# In[ ]:


df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mean())


# In[ ]:


df2=df.select_dtypes(exclude=object)
df2=pd.DataFrame(df2)


# In[ ]:


df2.isnull().sum()   ## checking null values of numerical type


# In[ ]:


df['GarageYrBlt'].sort_values(ascending=False).head()   ###No outliers


# In[ ]:


df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())


# In[ ]:


df3=df.select_dtypes(include=object)
df3=pd.DataFrame(df3)


# In[ ]:


df3.isnull().sum()      ### checking for null values of numerical type


# In[ ]:


for x in df.select_dtypes(include=object):
    df[x]=df[x].fillna(df[x].mode()[0])


# In[ ]:


df4=df.select_dtypes(include=object)
df4=pd.DataFrame(df4)


# ## Feature Engineering

# In[ ]:


df2=df.select_dtypes(exclude=object)
df2=pd.DataFrame(df2)


# In[ ]:


df2.head(10)


# In[ ]:


df4.head()


# In[ ]:


df4=pd.get_dummies(columns=df4.columns,drop_first=True,data=df4)


# In[ ]:


df4.head()


# In[ ]:


df=pd.concat([df2,df4],axis=1)


# In[ ]:


df.head()


# ## Model Training

# In[ ]:


from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV,train_test_split


# In[ ]:


features=df.drop(['Id','SalePrice'],axis=1)
target=df.SalePrice


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2)


# In[ ]:


pipelines={'lasso':make_pipeline(StandardScaler(),Lasso(random_state=123)),
          'ridge':make_pipeline(StandardScaler(),Ridge(random_state=123)),
          'enet':make_pipeline(StandardScaler(),ElasticNet(random_state=123)),
          'rf':make_pipeline(StandardScaler(),RandomForestRegressor(random_state=123)),
          'gb':make_pipeline(StandardScaler(),GradientBoostingRegressor(random_state=123))}


# In[ ]:


lasso_hyperparameters={'lasso__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]}
ridge_hyperparameters={'ridge__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]}
enet_hyperparameters={'elasticnet__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10],
                     'elasticnet__l1_ratio':[0.1,0.3,0.5,0.7,0.9]}
rf_hyperparameters={'randomforestregressor__n_estimators':[100,200],
                      'randomforestregressor__max_features':['auto','sqrt',0.33]}
gb_hyperparameters={'gradientboostingregressor__n_estimators':[100,200],
                   'gradientboostingregressor__learning_rate':[0.005,0.1,0.2],
                   'gradientboostingregressor__max_depth':[1,3,5]}


# In[ ]:


hyperparameters={'lasso':lasso_hyperparameters,
                'ridge':ridge_hyperparameters,
                'enet':enet_hyperparameters,
                'rf':rf_hyperparameters,
                'gb':gb_hyperparameters}


# In[ ]:





# In[ ]:





# In[ ]:


fitted_models={}
for name,pipeline in pipelines.items():
    model=GridSearchCV(pipeline,hyperparameters[name],cv=10,n_jobs=-1)
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    fitted_models[name]=model
    print(name,'has been fitted')


# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score


# In[ ]:


for name,model in fitted_models.items():
    pred=model.predict(x_test)
    print(name,'R2-',r2_score(y_test,pred))
    print('MeanSquaredError-',mean_squared_error(y_test,pred))
   


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




