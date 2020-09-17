#!/usr/bin/env python
# coding: utf-8

# # Colombia: Gender and income.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as sys


# ## Building the dataframe.

# In[2]:


people=pd.read_excel('final.xlsx')


# In[3]:


people.index=[people['DIRECTORIO_PER'],people['DIRECTORIO_HOG']]
people=people.drop(columns={'DIRECTORIO_HOG','DIRECTORIO_PER'})


# In[4]:


data=pd.read_excel('trabajo.xlsx')
data=data.drop(data[data['NPCKP24']==' '].index)
data.index=[data['DIRECTORIO_PER'],data['DIRECTORIO_HOG']]
data=data.drop(columns={'DIRECTORIO_HOG','DIRECTORIO_PER'})


# In[5]:


data=data.rename(columns={'NPCKP1':'Trabaja','NPCKP24':'Extra'})
data.head()


# In[6]:


people=people.merge(data,how='inner',right_index=True,left_index=True)
people.head()


# In[7]:


people=people.drop(people[people['Extra']==9].index)
people['Extra']=people.Extra.replace({2:0})


# In[8]:


people=people.drop(people[people['Gender']=='Intersex'].index)


# In[13]:


people.Gender=people.Gender.replace({'Female':1,'Male':0})


# In[17]:


people['Gender']=pd.Categorical(people['Gender'],categories=[0,1])


# In[19]:


people['Wage_m']=people['Wage_m'].replace({' ': np.NaN})


# In[20]:


people=people.dropna()


# In[21]:


people.head()


# In[34]:


ocup={1:'Working', 2: 'Looking for job', 3:
'Studying', 4 :'Housewives', 5:
'Incapacitado(a) permanente para trabajar', 6: 'Other activity'}


# In[10]:


plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.distplot(people.Wage_m)
plt.axvline(np.mean(people.Wage_m),linestyle ='--',color='red')
plt.title('Distribution of monthly wages\nIn Colombian pesos, 2017')
plt.xlabel('Wage\n(Mean: '+ str(1593472)+' pesos)')
plt.savefig('wage.png')


# ## Gender and income, F test.

# In[22]:


modelo1=sm.OLS.from_formula('Wage_m~Gender',data=people).fit()
modelo1.summary()


# In this part, the hypothesis tests developed under the ANOVA analysis of variance, can say that there is evidence that in Colombia women and men there is a great difference in relation to the salary that each one earns from their jobs, but it can be said that it is in all activities, such as at work or when studying and working at the same time?

# In[33]:


tabla=people.groupby('Gender').agg({'Wage_m': np.mean})
tabla.index=['Male','Female']

tabla


# The difference between the average salary of men and women is two hundred thousand Colombian pesos, although the difference between men is not great if they earn more than women, and it is the difference that the test of the regression shows.

# ### By occupation
# 
# 
#  * Working
#  * Looking for job
#  * Studying
#  * Housewives
#  * Other activity

# In[36]:


tab=pd.crosstab(people.Gender,people.Trabaja)
sumtab=tab.sum(axis=0)
tab1=tab/sumtab
tab1=tab1.rename(columns=ocup)
tab1.index=['Male','Female']
tab1


# Many of the occupations related to the maintenance of the home for men are more than half, while the housework for women has a large share.

# **Working**

# In[41]:


muestra1=people[people['Trabaja']==1]


# In[42]:


modelo2=sm.OLS.from_formula('Wage_m~Gender',data=muestra1).fit()
modelo2.summary()


# For people who work, earnings or salary does depend on gender, this is because with a significance level of 5%, the null hypothesis in which the mean salary received by women is equal to the mean can be rejected. of the salary that men receive.

# **Looking for job**

# In[43]:


muestra2=people[people['Trabaja']==2]


# In[44]:


modelo3=sm.OLS.from_formula('Wage_m~Gender',data=muestra2).fit()
modelo3.summary()


# For people who looking for work, earnings or salary does depend on gender, this is because with a significance level of 10%, the null hypothesis in which the mean salary received by women is equal to the mean can be rejected. of the salary that men receive.
# 

# **Studying**

# In[46]:


muestra3=people[people['Trabaja']==3]


# In[47]:


modelo4=sm.OLS.from_formula('Wage_m~Gender',data=muestra3).fit()
modelo4.summary()


# For people who study, earnings or salary does not depend on gender, this is because with a significance level of 5%, the null hypothesis in which the mean salary received by women is equal to the null hypothesis cannot be rejected. average salary received by men.

# **Housewives**

# In[48]:


muestra4=people[people['Trabaja']==4]


# In[49]:


modelo5=sm.OLS.from_formula('Wage_m~Gender',data=muestra4).fit()
modelo5.summary()


# For people who are housewives, earnings or salary depends on gender, this is because with a significance level of 5%, the null hypothesis in which the mean salary received by women is equal to the average salary received by men.

# **Other activity**

# In[50]:


muestra5=people[people['Trabaja']==6]


# In[51]:


modelo6=sm.OLS.from_formula('Wage_m~Gender',data=muestra5).fit()
modelo6.summary()


# For people who perform other activities, earnings or salary does not depend on gender, this is because with a significance level of 5%, the null hypothesis in which the mean salary received by women is the same cannot be rejected to the average salary that men receive.

# ## Conclution

# When looking at the effect of gender on income, it can be concluded that there are differences between men and women. On the other hand, when we look at these effects by economic activity that the person performs home chores or housewives, they show the dispersion between the salaries of women compared to men. When looking at work or looking for one, this difference of being a man or a woman is also visible. But when we look at the people who study or who carry out other economic activity, there are no differences between men and women in terms of salary.

# In[ ]:




