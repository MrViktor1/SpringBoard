#!/usr/bin/env python
# coding: utf-8

# # Springboard Time Series - 'Cowboy Cigarettes' Case Study - Tier 3

# ## Brief
# 
# You're working in the US federal government as a data scientist in the Health and Environment department. You've been tasked with determining whether sales for the oldest and most powerful producers of cigarettes in the country are increasing or declining. 
# 
# **Cowboy Cigarettes (TM, *est.* 1890)** is the US's longest-running cigarette manufacturer. Like many cigarette companies, however, they haven't always been that public about their sales and marketing data. The available post-war historical data runs for only 11 years after they resumed production in 1949; stopping in 1960 before resuming again in 1970. Your job is to use the 1949-1960 data to predict whether the manufacturer's cigarette sales actually increased, decreased, or stayed the same. You need to make a probable reconstruction of the sales record of the manufacturer - predicting the future, from the perspective of the past - to contribute to a full report on US public health in relation to major cigarette companies. 
# 
# The results of your analysis will be used as part of a major report relating public health and local economics, and will be combined with other studies executed by your colleagues to provide important government advice.  
# 
# -------------------------------
# As ever, this notebook is **tiered**, meaning you can elect that tier that is right for your confidence and skill level. There are 3 tiers, with tier 1 being the easiest and tier 3 being the hardest.  
# 
# **1. Sourcing and loading** 
# - Load relevant libraries 
# - Load the data
# - Explore the data
# 
#  
# **2. Cleaning, transforming and visualizing**
# - Dropping unwanted columns
# - Nomenclature
# - Type conversions
# - Making a predictor variable `y` 
# - Getting summary statistics for `y`
# - Plotting `y`
#   
#   
# **3. Modelling** 
# - Decomposition
#     - Trend
#     - Seasonality
#     - Noise
# - Testing for stationarity with KPSS
# - Making the data stationary
# - The ARIMA Model
#     - Make a function to find the MSE of a single ARIMA model
#     - Make a function to evaluate the different ARIMA models with different p, d, and q values
# - Visualize the results
# - Application: Forecasting
# 
# **4. Evaluating and concluding** 
# - What is our conclusion?
# - Next steps
#     

# ## 0. Preliminaries 
# 
# Time series data is just any data displaying how a single variable changes over time. It comes as a collection of metrics typically taken at regular intervals. Common examples of time series data include weekly sales data and daily stock prices. You can also easily acquire time series data from [Google Trends](https://trends.google.com/trends/?geo=US), which shows you how popular certain search terms are, measured in number of Google searches. 

# ## 1. Sourcing and Loading
# 
# ### 1a. Load relevant libraries 

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1b. Load the data
# Call the variable `cigData`. 

# In[95]:


cigData = pd.read_csv('CowboyCigsData.csv')


# ### 1c. Explore the data
# We now need to check whether the data conduces to a time series style analysis.

# In[96]:


cigData.head()


# Over a million cigarettes sold in the month of January 1949. This certainly is a popular cigarette brand. 

# Check out the columns feature of the data. How many columns are there? 

# In[97]:


cigData.columns
cigData.shape


# Let's check out the data types of our columns.

# In[98]:


cigData.dtypes


# Check whether there are any null values. 

# In[99]:


cigData.isnull().values.any()


# ## 2. Cleaning, transforming and visualizing

# # 2a. Dropping unwanted columns
# We need to cut that `Unnamed: 0` column. Delete it here.

# In[100]:


cigData.drop(columns = 'Unnamed: 0', inplace=True)


# ### 2b. Nomenclature

# We can see that the `Time` column actually has the granularity of months. Change the name of that column to `Month`.

# In[101]:


cigData.rename(columns = {'Time': 'Month'}, inplace=True)


# Call a head() to check this has worked. 

# In[102]:


cigData.head()


# In[ ]:


_ _ _


# ### 2c. Type conversions 

# Now, do time series analysis on a Pandas dataframe is overkill, and is actually counter-productive. It's much more easy to carry out this type of analysis if we convert our data to a series first.
# 
# Notice that the `Month` field was an object. Let's type convert the `Month` column to a Python `datetime`, before making that the index.

# In[103]:


cigData['Month'] = pd.to_datetime(cigData['Month'])
cigData.set_index('Month', inplace = True)


# Perfect! 

# ### 2d. Making a predictor variable `y`

# The data is now indexed by date, as time series data ought to be.
# 
# Since we want to predict the number of cigarette sales at Cowboy cigarettes, and `y` is typically used to signify a predictor variable, let's create a new variable called `y` and assign the indexed #Passenger column. 

# In[104]:


y = cigData['#CigSales']


# Check the type of our new variable. 

# In[105]:


type(y)


# ### 2e. Getting summary statistics for `y`

# Get the summary statistics of our data here. 

# In[106]:


cigData.describe()


# Try visualizing the data. A simple `matplotlib` plot should do the trick.  

# ### 2f. Plotting `y`

# In[107]:


y.plot()


# ## 3. Modelling 
# ### 3a. Decomposition
# What do you notice from the plot? Take at least `2` minutes to examine the plot, and write down everything you observe.  
# 
# All done?
# 
# We can see that, generally, there is a trend upwards in cigarette sales from at Cowboy Cigarettes. But there are also some striking - and perhaps unexpected - seasonal fluctuations. These seasonal fluctations come in a repeated pattern. Work out when these seasonal fluctuations are happening, and take 2 minutes to hypothesize on their cause here.
# 
# What does it mean to *decompose* time series data? It means breaking that data into 3 components: 
# 
# 1. **Trend**: The overall direction that the data is travelling in (like upwards or downwards)
# 2. **Seasonality**: Cyclical patterns in the data 
# 3. **Noise**: The random variation in the data
# 
# We can treat these components differently, depending on the question and what's appropriate in the context. They can either be added together in an *additive* model, or multiplied together in a *multiplicative* model. 
# 
# Make a coffee, take `5` minutes and read [this article](https://medium.com/@sigmundojr/seasonality-in-python-additive-or-multiplicative-model-d4b9cf1f48a7) and think about whether our data would conduce to an additive or multiplicative model here. Write your conclusion down just here: 
# 
# MULTIPLICATIVE MODEL MAKES SENSE AS THE MAGNITUDE OF THE SEASONALITY GROWS AS THE DATA INCREASES

# All done? Well, just on the basis of the plot above, it seems our Cowboy Cigarettes data is actually multiplicative. 
# 
# That's because, as time progresses, the general trend seems to be increasing *at a rate that's also increasing*. We also see that the seasonal fluctuations (the peaks and troughs) get bigger and bigger as time progresses.
# 
# Now on the other hand, if the data were simply additive, we could expect the general trend to increase at a *steadily*, and a constant speed; and also for seasonal ups and downs not to increase or decrease in extent over time.
# 
# Happily, we can use the `decompose()` function to quantify the component parts described above in our data.

# In[108]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(y)

trend = decomposition.trend
seasonal = decomposition.seasonal
resid = decomposition.resid
# Plot the original data, the trend, the seasonality, and the residuals 

plt.subplot(411)
plt.plot(y, label = 'best')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label = 'trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal, label = 'seasonal')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(resid, label = 'resid')
plt.legend(loc = 'best')
plt.tight_layout()


# ### 3b. Testing for stationarity with KPSS
# As you know, when doing time series analysis we always have to check for stationarity. Imprecisely, a time series dataset is stationary just if its statistical features don't change over time. A little more precisely, a stationary time series dataset will have constant mean, variance, and covariance.
# 
# There are many ways to test for stationarity, but one of the most common is the KPSS test. The Null hypothesis of this test is that the time series data in question is stationary; hence, if the *p*-value is less than the significance level (typically 0.05, but we decide) then we reject the Null and infer that the data is not stationary.

# In[109]:


from statsmodels.tsa.stattools import kpss

kpss(y)


# Since our p-value is less than 0.05, we should reject the Null hypothesis and deduce the non-stationarity of our data. 
# 
# But our data need to be stationary! So we need to do some transforming.

# ### 3c. Making the data stationary 
# Let's recall what it looks like. 

# In[110]:


y.plot()


# In our plot, we can see that both the mean and the variance *increase as time progresses*. At the moment, our data has neither a constant mean, nor a constant variance (the covariance, however, seems constant). 
# 
# One ofte  used way of getting rid of changing variance is to take the natural log of all the values in our dataset. Let's do this now. 

# In[111]:


y_log = np.log(y)


# 
# When you plot this, you can see how the variance in our data now remains contant over time.

# In[112]:


y_log.plot()


# We now have a constant variance, but we also need a constant mean.
# 
# We can do this by *differencing* our data. We difference a time series dataset when we create a new time series comprising the difference between the values of our existing dataset.
# 
# Python is powerful, and we can use the `diff()` function to do this. You'll notice there's one less value than our existing dataset (since we're taking the difference between the existing values).

# In[113]:


kpss(y_log.diff().dropna())


# Our p-value is now greater than 0.05, so we can accept the null hypothesis that our data is stationary.

# ### 3d. The ARIMA model
# 
# Recall that ARIMA models are based around the idea that it's possible to predict the next value in a time series by using information about the most recent data points. It also assumes there will be some randomness in our data that can't ever be predicted.
# 
# We can find some good parameters for our model using the `sklearn` and `statsmodels` libraries, and in particular `mean_squared_error` and `ARIMA`. 

# In[123]:


# Import mean_squared_error and ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


# #### 3di. Make a function to find the MSE of a single ARIMA model
# Things get intricate here. Don't worry if you can't do this yourself and need to drop down a Tier. 

# In[115]:


def eval_arima(data, arima_order):
    # Needs to be an integer because it is later used as an index.
    # Use int()
    split=int(len(data) * 0.8) 
    # Make train and test variables, with 'train, test'
    train, test = data[0:split], data[split:len(data)]
    past=[x for x in train]
    # make predictions
    predictions = list()
    for i in range(len(test)):#timestep-wise comparison between test data and one-step prediction ARIMA model. 
        model = ARIMA(past, order=arima_order)
        model_fit = model.fit(disp=0)
        future = model_fit.forecast()[0]
        predictions.append(future)
        past.append(test[i])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    # Return the error
    return error


# #### 3dii. Make a function to evaluate the different ARIMA models with different p, d, and q values

# In[116]:


def eval_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    # Iterate through p_values
    for p in p_values:
        # Iterate through d_values
        for d in d_values:
            # Iterate through q_values
            for q in q_values:
                # p, d, q iterator variables in that order
                order = (p,d,q)
                try:
                    # Make a variable called mse for the Mean squared error
                    mse = eval_arima(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    return print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# In[117]:


# Now, we choose a couple of values to try for each parameter.
p_values = [x for x in range(0,3)]
d_values = [x for x in range(0,3)]
q_values = [x for x in range(0,3)]


# In[118]:


# Finally, we can find the optimum ARIMA model for our data.
# Nb. this can take a while...!
import warnings
eval_models(y_log, p_values, d_values, q_values)


# So the best p,d, q, parameters for our ARIMA model are 2, 1, 1 respectively. Now we know this, we can build the model.

# In[124]:


p=2
d=1
q=3
model = ARIMA(y_log, order=(p,d,q))
model_fit = model.fit()
forecast = model_fit.forecast(24)


# We can take a look at a summary of the model this library has built around our data.

# In[125]:


model_fit.summary()


# ### 3e. Visualize the results 
# 
# Visualize the original dataset plotted against our model. 

# In[131]:


plt.figure(figsize=(15,10))
plt.plot(y_log.diff())
plt.plot(model_fit.predict(), color = 'red')


# ### 3f. Application: Forecasting
# 
# We've done well: our model fits pretty closely to our existing data. Let's now use it to forecast what's likely to occur in future.

# In[132]:


# Declare a variable called forecast_period with the amount of months to forecast, and
# create a range of future dates that is the length of the periods you've chosen to forecast
forecast_period = 24
date_range = pd.date_range(y_log.index[-1], periods = forecast_period, 
              freq='MS').strftime("%Y-%m-%d").tolist()

# Convert that range into a dataframe that includes your predictions
# First, call DataFrame on pd
future_months = pd.DataFrame(date_range, columns = ['Month'])
# Let's now convert the 'Month' column to a datetime object with to_datetime 
future_months['Month'] = pd.to_datetime(future_months['Month'])
future_months.set_index('Month', inplace = True)
future_months['Prediction'] = forecast[0]

# Plot your future predictions
# Call figure() on plt
plt.figure(figsize=(15,10))
plt.plot(y_log)
plt.plot(y_log['Nov 1960'].append(future_months['Prediction']))
plt.show()


# ## 4. Evaluating and Concluding
# 
# Our model captures the centre of a line that's increasing at a remarkable rate. Cowboy Cigarettes sell more cigarettes in the summer, perhaps due to the good weather, disposable income and time off that people enjoy, and the least in the winter, when people might be spending less and enjoying less free time outdoors. 
# 
# Remarkably, our ARIMA model made predictions using just one variable. We can only speculate, however, on the causes of the behaviour predicted by our model. We should also take heed that spikes in data, due to sudden unusual circumstances like wars, are not handled well by ARIMA; and the outbreak of the Vietnam War in the 1960s would likely cause our model some distress.  
# 
# We could suggest to our employers that, if they are interested in discovering the causes of the cigarette sales trajectory, they execute a regression analysis in addition to the time series one. 
