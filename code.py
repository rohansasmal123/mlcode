#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[2]:


datasets = pandas.read_csv('SalaryData.csv')


# In[3]:


datasets.head()


# In[4]:


x = datasets['YearsExperience'].values.reshape(30,1)


# In[5]:


y = datasets['Salary']


# In[6]:


#import matplotlib.pyplot as plt
#import seaborn as sb


# In[7]:


#plt.scatter(x,y)

#plt.xlabel('Experience')
#plt.ylabel('Salary')

#plt.plot(x,y,marker='o')


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


mind = LinearRegression()


# In[10]:


from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# In[ ]:





# In[11]:


mind.fit(X_train,y_train)


# In[12]:


y_pred = mind.predict(X_test)


# In[13]:


y_pred


# In[14]:


y_test


# In[15]:


#plt.scatter(X_test, y_test)
#plt.scatter(X_test, y_pred, color='red')


# In[16]:


from sklearn import metrics


# In[18]:


metrics.mean_squared_error(y_test,y_pred)


# In[19]:


mind.intercept_


# In[20]:


mind.predict([[2.2]])


# ## 

# In[21]:


from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)

file1 = open("accuracy.txt", "w")
file1.write("{0}".format(score))
file1.close()



