#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')


# In[6]:


wine = pd.read_csv("winequality-red.csv")
print("Successfully Imported Data!")
wine.head()


# In[7]:


print(wine.shape)


# In[8]:


wine.describe(include='all')


# In[9]:


print(wine.isna().sum())


# In[10]:


wine.corr()


# In[11]:


wine.groupby('quality').mean()


# In[12]:


import seaborn as sns


# In[14]:


data='wine'


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[21]:


data='wine'
x='quality'


# In[22]:


sns.countplot(data=wine, x='quality')
plt.show()


# In[26]:


data='wine'
x='pH'

sns.countplot(data=wine, x='pH')
plt.show()


# In[27]:


sns.countplot(data=wine, x='alcohol')
plt.show()


# In[28]:


sns.countplot(data=wine, x='fixed acidity')
plt.show()


# In[29]:


sns.countplot(data=wine,x='volatile acidity')
plt.show()


# In[31]:


sns.countplot(data=wine, x='citric acid')
plt.show()


# In[32]:


sns.countplot(data=wine,x='density')
plt.show()


# In[33]:


sns.kdeplot(wine.query('quality > 2').quality)


# In[34]:


sns.distplot(wine['alcohol'])


# In[35]:


wine.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)


# In[36]:


wine.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)


# In[37]:


wine.hist(figsize=(10,10),bins=50)
plt.show()


# In[38]:


corr = wine.corr()
sns.heatmap(corr,annot=True)


# In[39]:


sns.pairplot(wine)


# In[40]:


sns.violinplot(x='quality', y='alcohol', data=wine)


# In[41]:


wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]# Separate feature variables and target variable
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']


# In[42]:


wine['goodquality'].value_counts()


# In[43]:


X


# In[44]:


print(Y)


# In[45]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)


# In[46]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))


# In[47]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','KNN', 'SVC','Decision Tree' ,'GaussianNB','Random Forest','Xgboost'],
    'Score': [0.870,0.872,0.868,0.864,0.833,0.893,0.879]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# In[ ]:




