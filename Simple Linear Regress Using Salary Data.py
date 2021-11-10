#!/usr/bin/env python
# coding: utf-8

# In[3]:


# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


# import dataset
dataset=pd.read_csv('C:\\Users\\Admin\\Downloads\\Salary_Data (3).CSV')
dataset       


# In[13]:


plt.hist(dataset.YearsExperience)
plt.boxplot(dataset.YearsExperience,1,"rs",1)


plt.hist(dataset.Salary)
plt.boxplot(dataset.Salary)

plt.plot(dataset.YearsExperience,dataset.Salary,"");plt.xlabel("YearsExperience");plt.ylabel("Salary")


dataset.Salary.corr(dataset.YearsExperience) # # correlation value between X and Y
np.corrcoef(dataset.Salary,dataset.YearsExperience)


# In[16]:


# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=dataset).fit()


# In[18]:


# For getting coefficients of the varibles used in equation
model.params


# In[20]:


# P-values for the variables and R-squared value for prepared model
model.summary()


# In[22]:


model.conf_int(0.05) # 95% confidence interval


# In[24]:


pred = model.predict(dataset.iloc[:,0]) # Predicted values of Salary using the model


# In[28]:


# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=dataset['YearsExperience'],y=dataset['Salary'],color='red');plt.plot(dataset['YearsExperience'],pred,color='black');plt.xlabel('Salary');plt.ylabel('TISSUE')

pred.corr(dataset.Sal) # 0.81


# In[30]:


# Transforming variables for accuracy
model2 = smf.ols('Salary~np.log(YearsExperience)',data=dataset).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(pd.DataFrame(dataset['YearsExperience']))
pred2.corr(dataset.Salary)
# pred2 = model2.predict(dataset.iloc[:,0])
pred2
plt.scatter(x=dataset['YearsExperience'],y=dataset['Salary'],color='green');plt.plot(dataset['YearsExperience'],pred2,color='blue');plt.xlabel('YearsExperience');plt.ylabel('TISSUE')


# In[32]:


# Exponential transformation
model3 = smf.ols('np.log(Salary)~YearsExperience',data=dataset).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(dataset['YearsExperience']))
pred_log
pred3=np.exp(pred_log)  # as we have used log(salaryy) in preparing model so we need to convert it back
pred3
pred3.corr(dataset.Salary)
plt.scatter(x=dataset['YearsExperience'],y=dataset['Salary'],color='green');plt.plot(dataset.YearsExperience,np.exp(pred_log),color='blue');plt.xlabel('YearsExperience');plt.ylabel('TISSUE')
resid_3 = pred3-dataset.Salary


# In[34]:


# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=dataset.Salary);plt.xlabel("Predicted");plt.ylabel("Actual")


# In[37]:


# Quadratic model
dataset["YearsExperience"] = dataset.YearsExperience*dataset.YearsExperience
model_quad = smf.ols("Salary~YearsExperience+YearsExperience",data=dataset).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(dataset.YearsExperience)

model_quad.conf_int(0.05) # 
plt.scatter(dataset.YearsExperience,dataset.Salary,c="b");plt.plot(dataset.YearsExperience,pred_quad,"r")

plt.scatter(np.arange(109),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 


# In[42]:


############################### Implementing the Linear Regression model from sklearn library

from sklearn.linear_model import LinearRegression
import numpy as np
plt.scatter(dataset.YearsExperience,dataset.Salary)
model1 = LinearRegression()
model1.fit(dataset.YearsExperience.values.reshape(-1,1),dataset.Salary)
pred1 = model1.predict(dataset.YearsExperience.values.reshape(-1,1))
# Adjusted R-Squared value
model1.score(dataset.YearsExperience.values.reshape(-1,1),dataset.Salary)# 0.6700
rmse1 = np.sqrt(np.mean((pred1-dataset.Salary)**2)) # 32.760
model1.coef_
model1.intercept_


# In[41]:


plt.hist(model_quad.resid_pearson) # histogram for residual values 


# In[44]:


############################### Implementing the Linear Regression model from sklearn library

from sklearn.linear_model import LinearRegression
import numpy as np
plt.scatter(dataset.YearsExperience,dataset.Salary)
model1 = LinearRegression()
model1.fit(dataset.YearsExperience.values.reshape(-1,1),dataset.Salary)
pred1 = model1.predict(dataset.YearsExperience.values.reshape(-1,1))
# Adjusted R-Squared value
model1.score(dataset.YearsExperience.values.reshape(-1,1),dataset.Salary)# 0.6700
rmse1 = np.sqrt(np.mean((pred1-dataset.Salary)**2)) # 32.760
model1.coef_
model1.intercept_


# In[47]:


#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred1,(pred1-dataset.Salary),c="r")
plt.hlines(y=0,xmin=0,xmax=300) 
# checking normal distribution for residual
plt.hist(pred1-dataset.Salary)


# In[49]:


### Fitting Quadratic Regression 
dataset["YearsExperience_sqrd"] = dataset.YearsExperience*dataset.YearsExperience
model2 = LinearRegression()
model2.fit(X = dataset.iloc[:,[0,2]],y=dataset.Salary)
pred2 = model2.predict(dataset.iloc[:,[0,2]])


# In[51]:


# Adjusted R-Squared value
model2.score(dataset.iloc[:,[0,2]],dataset.Salary)# 0.67791
rmse2 = np.sqrt(np.mean((pred2-dataset.Salary)**2)) # 32.366
model2.coef_
model2.intercept_


# In[53]:


#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred2,(pred2-dataset.Salary),c="r")
plt.hlines(y=0,xmin=0,xmax=200) 


# In[55]:


# Checking normal distribution
plt.hist(pred2-dataset.Salary)
import pylab
import scipy.stats as st
st.probplot(pred2-dataset.Salary,dist="norm",plot=pylab)


# In[59]:


# Let us prepare a model by applying transformation on dependent variable
dataset["Salary_sqrt"] = np.sqrt(dataset.Salary)

model3 = LinearRegression()
model3.fit(X = dataset.iloc[:,[0,2]],y=dataset.Salary_sqrt)
pred3 = model3.predict(dataset.iloc[:,[0,2]])


# In[61]:


model3.score(dataset.iloc[:,[0,2]],dataset.Salary_sqrt)# 0.74051
rmse3 = np.sqrt(np.mean(((pred3)**2-dataset.Salary)**2)) # 32.0507
model3.coef_
model3.intercept_
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred3)**2,((pred3)**2-dataset.Salary),c="r")
plt.hlines(y=0,xmin=0,xmax=300)  
# checking normal distribution for residuals 
plt.hist((pred3)**2-dataset.Salary)
st.probplot((pred3)**2-dataset.Salary,dist="norm",plot=pylab)

# Let us prepare a model by applying transformation on dependent variable without transformation on input variables 
model4 = LinearRegression()
model4.fit(X = dataset.YearsExperience.values.reshape(-1,1),y=dataset.Salary_sqrt)
pred4 = model4.predict(dataset.YearsExperience.values.reshape(-1,1))
# Adjusted R-Squared value
model4.score(dataset.YearsExperience.values.reshape(-1,1),dataset.Salary_sqrt)# 0.7096
rmse4 = np.sqrt(np.mean(((pred4)**2-dataset.Salary)**2)) # 34.165
model4.coef_
model4.intercept_
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred4)**2,((pred4)**2-dataset.Salary),c="r")
plt.hlines(y=0,xmin=0,xmax=300)  

st.probplot((pred4)**2-dataset.Salary,dist="norm",plot=pylab)

# Checking normal distribution for residuals 
plt.hist((pred4)**2-dataset.Salary)

