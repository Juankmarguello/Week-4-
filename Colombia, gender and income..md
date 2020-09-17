# Colombia: Gender and income.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as sys
```

## Building the dataframe.


```python
people=pd.read_excel('final.xlsx')
```


```python
people.index=[people['DIRECTORIO_PER'],people['DIRECTORIO_HOG']]
people=people.drop(columns={'DIRECTORIO_HOG','DIRECTORIO_PER'})
```


```python
data=pd.read_excel('trabajo.xlsx')
data=data.drop(data[data['NPCKP24']==' '].index)
data.index=[data['DIRECTORIO_PER'],data['DIRECTORIO_HOG']]
data=data.drop(columns={'DIRECTORIO_HOG','DIRECTORIO_PER'})

```


```python
data=data.rename(columns={'NPCKP1':'Trabaja','NPCKP24':'Extra'})
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Trabaja</th>
      <th>Extra</th>
    </tr>
    <tr>
      <th>DIRECTORIO_PER</th>
      <th>DIRECTORIO_HOG</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10100012</th>
      <th>1010001</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10100013</th>
      <th>1010001</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10100111</th>
      <th>1010011</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10100212</th>
      <th>1010021</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10100312</th>
      <th>1010031</th>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
people=people.merge(data,how='inner',right_index=True,left_index=True)
people.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Age</th>
      <th>Marital</th>
      <th>Gender</th>
      <th>Wage_m</th>
      <th>Trabaja</th>
      <th>Extra</th>
    </tr>
    <tr>
      <th>DIRECTORIO_PER</th>
      <th>DIRECTORIO_HOG</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12401712</th>
      <th>1240171</th>
      <td>37</td>
      <td>2</td>
      <td>Female</td>
      <td>50000</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>24022513</th>
      <th>2402251</th>
      <td>29</td>
      <td>5</td>
      <td>Female</td>
      <td>1700000</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28212411</th>
      <th>2821241</th>
      <td>38</td>
      <td>2</td>
      <td>Male</td>
      <td>950000</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11220211</th>
      <th>1122021</th>
      <td>53</td>
      <td>5</td>
      <td>Female</td>
      <td>5600000</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14438915</th>
      <th>1443891</th>
      <td>22</td>
      <td>5</td>
      <td>Female</td>
      <td>700000</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
people=people.drop(people[people['Extra']==9].index)
people['Extra']=people.Extra.replace({2:0})
```


```python
people=people.drop(people[people['Gender']=='Intersex'].index)
```


```python
people.Gender=people.Gender.replace({'Female':1,'Male':0})
```


```python
people['Gender']=pd.Categorical(people['Gender'],categories=[0,1])

```


```python
people['Wage_m']=people['Wage_m'].replace({' ': np.NaN})
```


```python
people=people.dropna()
```


```python
people.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Age</th>
      <th>Marital</th>
      <th>Gender</th>
      <th>Wage_m</th>
      <th>Trabaja</th>
      <th>Extra</th>
    </tr>
    <tr>
      <th>DIRECTORIO_PER</th>
      <th>DIRECTORIO_HOG</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24022513</th>
      <th>2402251</th>
      <td>29</td>
      <td>5</td>
      <td>1</td>
      <td>1700000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28212411</th>
      <th>2821241</th>
      <td>38</td>
      <td>2</td>
      <td>0</td>
      <td>950000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11220211</th>
      <th>1122021</th>
      <td>53</td>
      <td>5</td>
      <td>1</td>
      <td>5600000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14438915</th>
      <th>1443891</th>
      <td>22</td>
      <td>5</td>
      <td>1</td>
      <td>700000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10339711</th>
      <th>1033971</th>
      <td>48</td>
      <td>6</td>
      <td>1</td>
      <td>735800</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
ocup={1:'Working', 2: 'Looking for job', 3:
'Studying', 4 :'Housewives', 5:
'Incapacitado(a) permanente para trabajar', 6: 'Other activity'}
```


```python
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.distplot(people.Wage_m)
plt.axvline(np.mean(people.Wage_m),linestyle ='--',color='red')
plt.title('Distribution of monthly wages\nIn Colombian pesos, 2017')
plt.xlabel('Wage\n(Mean: '+ str(1593472)+' pesos)')
plt.savefig('wage.png')
```


![png](output_16_0.png)


## Gender and income, F test.


```python
modelo1=sm.OLS.from_formula('Wage_m~Gender',data=people).fit()
modelo1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Wage_m</td>      <th>  R-squared:         </th>  <td>   0.002</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.002</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   210.6</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 16 Sep 2020</td> <th>  Prob (F-statistic):</th>  <td>1.17e-47</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>23:29:12</td>     <th>  Log-Likelihood:    </th> <td>-1.4805e+06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 92785</td>      <th>  AIC:               </th>  <td>2.961e+06</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 92783</td>      <th>  BIC:               </th>  <td>2.961e+06</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td> 1.689e+06</td> <td> 9424.661</td> <td>  179.169</td> <td> 0.000</td> <td> 1.67e+06</td> <td> 1.71e+06</td>
</tr>
<tr>
  <th>Gender[T.1]</th> <td>-1.963e+05</td> <td> 1.35e+04</td> <td>  -14.510</td> <td> 0.000</td> <td>-2.23e+05</td> <td> -1.7e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>95612.088</td> <th>  Durbin-Watson:     </th>  <td>   2.000</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>7099171.486</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 5.142</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>44.600</td>   <th>  Cond. No.          </th>  <td>    2.59</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



In this part, the hypothesis tests developed under the ANOVA analysis of variance, can say that there is evidence that in Colombia women and men there is a great difference in relation to the salary that each one earns from their jobs, but it can be said that it is in all activities, such as at work or when studying and working at the same time?


```python
tabla=people.groupby('Gender').agg({'Wage_m': np.mean})
tabla.index=['Male','Female']

tabla
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wage_m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male</th>
      <td>1.688612e+06</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>1.492351e+06</td>
    </tr>
  </tbody>
</table>
</div>



The difference between the average salary of men and women is two hundred thousand Colombian pesos, although the difference between men is not great if they earn more than women, and it is the difference that the test of the regression shows.

### By occupation


 * Working
 * Looking for job
 * Studying
 * Housewives
 * Other activity


```python
tab=pd.crosstab(people.Gender,people.Trabaja)
sumtab=tab.sum(axis=0)
tab1=tab/sumtab
tab1=tab1.rename(columns=ocup)
tab1.index=['Male','Female']
tab1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Trabaja</th>
      <th>Working</th>
      <th>Looking for job</th>
      <th>Studying</th>
      <th>Housewives</th>
      <th>Other activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male</th>
      <td>0.519757</td>
      <td>0.542857</td>
      <td>0.496395</td>
      <td>0.148054</td>
      <td>0.420048</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>0.480243</td>
      <td>0.457143</td>
      <td>0.503605</td>
      <td>0.851946</td>
      <td>0.579952</td>
    </tr>
  </tbody>
</table>
</div>



Many of the occupations related to the maintenance of the home for men are more than half, while the housework for women has a large share.

**Working**


```python
muestra1=people[people['Trabaja']==1]
```


```python
modelo2=sm.OLS.from_formula('Wage_m~Gender',data=muestra1).fit()
modelo2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Wage_m</td>      <th>  R-squared:         </th>  <td>   0.002</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.002</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   163.7</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 17 Sep 2020</td> <th>  Prob (F-statistic):</th>  <td>1.87e-37</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>00:07:01</td>     <th>  Log-Likelihood:    </th> <td>-1.4318e+06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 89688</td>      <th>  AIC:               </th>  <td>2.864e+06</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 89686</td>      <th>  BIC:               </th>  <td>2.864e+06</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td> 1.709e+06</td> <td> 9606.822</td> <td>  177.887</td> <td> 0.000</td> <td> 1.69e+06</td> <td> 1.73e+06</td>
</tr>
<tr>
  <th>Gender[T.1]</th> <td>-1.774e+05</td> <td> 1.39e+04</td> <td>  -12.796</td> <td> 0.000</td> <td>-2.05e+05</td> <td> -1.5e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>92237.113</td> <th>  Durbin-Watson:     </th>  <td>   2.002</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>6765176.469</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 5.129</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>44.293</td>   <th>  Cond. No.          </th>  <td>    2.57</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



For people who work, earnings or salary does depend on gender, this is because with a significance level of 5%, the null hypothesis in which the mean salary received by women is equal to the mean can be rejected. of the salary that men receive.

**Looking for job**


```python
muestra2=people[people['Trabaja']==2]
```


```python
modelo3=sm.OLS.from_formula('Wage_m~Gender',data=muestra2).fit()
modelo3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Wage_m</td>      <th>  R-squared:         </th> <td>   0.007</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.005</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3.739</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 17 Sep 2020</td> <th>  Prob (F-statistic):</th>  <td>0.0537</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>00:07:25</td>     <th>  Log-Likelihood:    </th> <td> -7607.0</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   525</td>      <th>  AIC:               </th> <td>1.522e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   523</td>      <th>  BIC:               </th> <td>1.523e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td> 4.171e+05</td> <td> 2.82e+04</td> <td>   14.805</td> <td> 0.000</td> <td> 3.62e+05</td> <td> 4.72e+05</td>
</tr>
<tr>
  <th>Gender[T.1]</th> <td>-8.058e+04</td> <td> 4.17e+04</td> <td>   -1.934</td> <td> 0.054</td> <td>-1.62e+05</td> <td> 1282.942</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>564.347</td> <th>  Durbin-Watson:     </th> <td>   2.142</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>41869.424</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 4.779</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>45.693</td>  <th>  Cond. No.          </th> <td>    2.53</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



For people who looking for work, earnings or salary does depend on gender, this is because with a significance level of 10%, the null hypothesis in which the mean salary received by women is equal to the mean can be rejected. of the salary that men receive.


**Studying**


```python
muestra3=people[people['Trabaja']==3]
```


```python
modelo4=sm.OLS.from_formula('Wage_m~Gender',data=muestra3).fit()
modelo4.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Wage_m</td>      <th>  R-squared:         </th> <td>   0.003</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.002</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2.595</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 17 Sep 2020</td> <th>  Prob (F-statistic):</th>  <td> 0.108</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>00:12:41</td>     <th>  Log-Likelihood:    </th> <td> -15062.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   971</td>      <th>  AIC:               </th> <td>3.013e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   969</td>      <th>  BIC:               </th> <td>3.014e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>   8.5e+05</td> <td> 6.02e+04</td> <td>   14.127</td> <td> 0.000</td> <td> 7.32e+05</td> <td> 9.68e+05</td>
</tr>
<tr>
  <th>Gender[T.1]</th> <td>-1.366e+05</td> <td> 8.48e+04</td> <td>   -1.611</td> <td> 0.108</td> <td>-3.03e+05</td> <td> 2.98e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1344.549</td> <th>  Durbin-Watson:     </th>  <td>   1.964</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>283611.728</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 7.605</td>  <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>85.332</td>  <th>  Cond. No.          </th>  <td>    2.63</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



For people who study, earnings or salary does not depend on gender, this is because with a significance level of 5%, the null hypothesis in which the mean salary received by women is equal to the null hypothesis cannot be rejected. average salary received by men.

**Housewives**


```python
muestra4=people[people['Trabaja']==4]
```


```python
modelo5=sm.OLS.from_formula('Wage_m~Gender',data=muestra4).fit()
modelo5.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Wage_m</td>      <th>  R-squared:         </th> <td>   0.018</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.017</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   21.17</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 17 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>4.65e-06</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>00:20:12</td>     <th>  Log-Likelihood:    </th> <td> -17959.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1182</td>      <th>  AIC:               </th> <td>3.592e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1180</td>      <th>  BIC:               </th> <td>3.593e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td> 8.086e+05</td> <td> 7.26e+04</td> <td>   11.135</td> <td> 0.000</td> <td> 6.66e+05</td> <td> 9.51e+05</td>
</tr>
<tr>
  <th>Gender[T.1]</th> <td> -3.62e+05</td> <td> 7.87e+04</td> <td>   -4.601</td> <td> 0.000</td> <td>-5.16e+05</td> <td>-2.08e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1606.925</td> <th>  Durbin-Watson:     </th>  <td>   2.076</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>295092.537</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 7.520</td>  <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>78.931</td>  <th>  Cond. No.          </th>  <td>    5.02</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



For people who are housewives, earnings or salary depends on gender, this is because with a significance level of 5%, the null hypothesis in which the mean salary received by women is equal to the average salary received by men.

**Other activity**


```python
muestra5=people[people['Trabaja']==6]
```


```python
modelo6=sm.OLS.from_formula('Wage_m~Gender',data=muestra5).fit()
modelo6.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Wage_m</td>      <th>  R-squared:         </th> <td>   0.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.002</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td> 0.05607</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 17 Sep 2020</td> <th>  Prob (F-statistic):</th>  <td> 0.813</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>00:22:06</td>     <th>  Log-Likelihood:    </th> <td> -6699.9</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   419</td>      <th>  AIC:               </th> <td>1.340e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   417</td>      <th>  BIC:               </th> <td>1.341e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td> 1.537e+06</td> <td> 1.61e+05</td> <td>    9.552</td> <td> 0.000</td> <td> 1.22e+06</td> <td> 1.85e+06</td>
</tr>
<tr>
  <th>Gender[T.1]</th> <td> 5.002e+04</td> <td> 2.11e+05</td> <td>    0.237</td> <td> 0.813</td> <td>-3.65e+05</td> <td> 4.65e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>337.899</td> <th>  Durbin-Watson:     </th> <td>   1.870</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>5663.478</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 3.424</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>19.658</td>  <th>  Cond. No.          </th> <td>    2.85</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



For people who perform other activities, earnings or salary does not depend on gender, this is because with a significance level of 5%, the null hypothesis in which the mean salary received by women is the same cannot be rejected to the average salary that men receive.

## Conclution

When looking at the effect of gender on income, it can be concluded that there are differences between men and women. On the other hand, when we look at these effects by economic activity that the person performs home chores or housewives, they show the dispersion between the salaries of women compared to men. When looking at work or looking for one, this difference of being a man or a woman is also visible. But when we look at the people who study or who carry out other economic activity, there are no differences between men and women in terms of salary.


```python

```
