#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


link='https://raw.githubusercontent.com/Sarani1507/Machine-Learning-and-Big-Data/main/seattle-weather.csv'
df = pd.read_csv(link)


# In[3]:


df.head()


# In[4]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isna().sum()


# In[9]:


sns.heatmap(df.corr(),annot=True)


# In[10]:


sns.countplot(data=df, x='weather')


# In[11]:


fig,axes = plt.subplots(2,2, figsize=(10,10))
cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
for i in range(4):
    sns.boxplot(x='weather', y=cols[i], data=df, ax=axes[i%2,i//2])


# In[12]:


sns.pairplot(data=df, hue='weather')


# In[13]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['weather'] = label_encoder.fit_transform(df['weather'])


# In[14]:


df.head()


# In[15]:


X = df.iloc[:,1:-1]
y = df.iloc[:,-1]


# In[16]:


y.head()


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.25)


# In[18]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit_transform(X_train)
std.fit_transform(X_test)


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[20]:


lr = LogisticRegression()
rf = RandomForestClassifier(bootstrap=False)
gbc = GradientBoostingClassifier()
dt = DecisionTreeClassifier()
svc = SVC()
knn= KNeighborsClassifier()


# In[44]:


lr.fit(X_train,y_train)
rf.fit(X_train,y_train)
gbc.fit(X_train,y_train)
dt.fit(X_train,y_train)
svc.fit(X_train, y_train)
knn.fit(X_train, y_train)


# In[45]:


y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gbc = gbc.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_knn = knn.predict(X_test)


# In[46]:


print('LogReg Accuracy = {:.2f}'.format(lr.score(X_test,y_test)))
print('RandFor Accuracy = {:.2f}'.format(rf.score(X_test,y_test)))
print('GBC Accuracy = {:.2f}'.format(gbc.score(X_test,y_test)))
print('DT Accuracy = {:.2f}'.format(dt.score(X_test,y_test)))
print('SVC Accuracy = {:.2f}'.format(svc.score(X_test,y_test)))
print('KNN Accuracy = {:.2f}'.format(knn.score(X_test,y_test)))


# In[47]:


a=['Logistic regression','Randomforest','Gradient boosting','Decision tree','SVC','KNN']
b=[lr.score(X_test,y_test),rf.score(X_test,y_test),gbc.score(X_test,y_test),dt.score(X_test,y_test),svc.score(X_test,y_test),knn.score(X_test,y_test)]


# In[48]:


sns.barplot(a,b)


# In[ ]:




