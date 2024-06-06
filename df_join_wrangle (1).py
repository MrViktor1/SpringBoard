#!/usr/bin/env python
# coding: utf-8

# In[11]:



#!pip uninstall numba
get_ipython().system('pip install numba==0.58.1')


# In[3]:


get_ipython().system('pip list --local > installed_packages.txt')


# In[6]:


get_ipython().system('pip install ydata-profiling')


# In[1]:


# import the needed packages to create a data frame and run a profile report 
from pathlib import Path

import requests
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model, preprocessing

from pandas_profiling.utils.cache import cache_file


# In[ ]:





# In[3]:


# assign file paths to variables
prospects_file = 'raw_data/nfl_draft_prospects.csv.zip'
profiles_file = 'raw_data/nfl_draft_profiles.csv.zip'

# use pandas read csv to create data frames 

prospects_df = pd.read_csv(prospects_file)
profiles_df = pd.read_csv(profiles_file)


# In[4]:


# merge data sets to make a complete data frame including both draft position and player description
complete_prospro = prospects_df.merge(profiles_df, how='left', on='player_id')


# In[5]:


# drop columns that are not of use in analysis
useless_columns = ['position_x', 'school_abbr_x', 'school_y', 'school_abbr_y', 'school_name_y', 'link_y', 
                   'school_logo', 'traded', 'link_x', 'weight_x', 'height_x', 'pos_rk_x', 'ovr_rk_x', 'player_image_x',
                  'guid_y', 'player_name_y', 'weight_y', 'height_y', 'player_image_y', 'pos_rk_y', 'ovr_rk_y', 'grade_y',
                  'team_abbr', 'team_logo_espn', 'position_y', 'trade_note', 'pos_abbr_y']
complete_prospro = complete_prospro.drop(useless_columns, axis=1)


# In[6]:


# subset the text columns and return a 0 for Nan and 1 for any other data type
with_text = pd.notna(complete_prospro[['text1', 'text2', 'text3', 'text4']])
# sum along the rows to see if the player observation contains values 
with_text = with_text.sum(axis=1)


# In[7]:


# drop rows for players that do not contain any text data
df = complete_prospro[with_text > 0]


# In[8]:


# drop players that were not drafted (do not have a value for pick)
df = df.dropna(subset='pick')


# In[10]:


report = df.profile_report(explorative=True, html={'style': {'full_width': True}})
report


# In[73]:


df['grade_x'].unique()


# 1. Initially I will be testing if player grades can be used to estimate player draft position better than random chance alone by using an r squared test on a model that gives the mean standard deviation from the sample to get and observed correlation statistic and then shuffling the scores using np.random_choice and reruning the sample 10000 times. I will then plot the distribution and see if the observed standard error p-value is extreme enough to land outside of the 95% confidence interval
# 
# 2. Second of all I would like to test if using a natural language processed score from my text data can be used to improve the predictive power of my model compared to using the player grade alone.  For this I will first conduct the same experiment as above to compare my pradictive variables mean squared error to a hacker sample made from shuffling and random sampling and then observing the p_value of my findings. 

# In[9]:


#combine text columns
df['text'] = df['text1'].fillna('')+df['text2'].fillna('')+df['text3'].fillna('')+df['text4'].fillna('')
#drop old text data columns
df=df.drop(columns=['text1','text2','text3','text4'])


# In[10]:


#drop rows from df where grades are nan 
df.dropna(subset='grade_x', inplace=True)


# In[16]:



sns.scatterplot(x=df['grade_x'], y=df['pick'])
plt.show()
# why are players with low grades being picked first overall? why does the y axis only go to 40 when the picks
#go well into the hundreds?
# Answer: must factor round into the pick number (this can be found in data set as 'overall')


# In[17]:


#make a histogram of the explanatory variable
sns.histplot(df['grade_x'], bins=40)
#appears that the lower end has a disproportional occurences, possibly due to lack of coverage/knowledge of these players
#clip lower values of grades column to avoid factoring innacurrate/lazy ratings into the model
df = df[df['grade_x']>50]


# In[27]:


#rerun histogram and scatter plot with the y value as overall
sns.scatterplot(x=df['grade_x'], y=df['overall'])
plt.show()

sns.histplot(df['grade_x'], bins=40)


# In[28]:


sns.heatmap(df.corr(), annot=True)

#are dependant variable overall has a strong pearson coefficient with the grade_x column which we expected to be true


# In[17]:


#plot a line of best fit to see where the correlation lies
sns.regplot(x=df['grade_x'], y=df['overall'], line_kws={"color": "red"})


# In[11]:


#make training and test data splits using sklearn
X = df[["grade_x"]] 
y = df[["overall"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123)


# In[19]:


rModel = linear_model.LinearRegression(normalize=True)

rModel_results = rModel.fit(X_train, y_train)


# In[31]:


y_pred = rModel.predict(X_test)


# In[ ]:


plt.scatter(y_test,y_pred)

print(rModel.score(X_train, y_train))


# The next step would be using nlp to create a metric for sentiment on how the players are talked about and use this in addition to the player grade to run a multiple linear regression model to potentially improve the predictive power of the model. I would train on historical data and use the current year draft as the test data.

# In[21]:


df.head()


# In[12]:


#get rid of non usefull columns
df = df.drop(['player_id', 'alt_player_id'], axis='columns')


# In[77]:


df.dtypes


# In[13]:


dfd=df[['pos_abbr_x', 'school_x', 'team']] # select object type columns
df = pd.concat([df.drop(dfd, axis=1), pd.get_dummies(dfd)], axis=1)


# In[14]:


#turn df bool into numeric values for analysis
dfnumeric = df.select_dtypes(include=['bool'])
df = pd.concat([df.drop(dfnumeric, axis=1), dfnumeric.astype(int)], axis=1 )


# In[16]:


df.head()


# In[19]:


# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler
MM_scaler = MinMaxScaler()

# Fit MM_scaler to the data
MM_scaler.fit(df[['draft_year','overall', 'grade_x']])

# Transform the data using the fitted scaler
df[['MM_draft_year','MM_overall', 'MM_grade_x']] = MM_scaler.transform(df[['draft_year','overall', 'grade_x']])

# Compare the origional and transformed column
print(df['MM_draft_year'].head(-5))


# In[20]:


df.head()


# In[21]:


#assign input columns to X and predicted y columns to y
X = df.drop(columns=['draft_year', 'player_name_x', 'school_name_x', 'pick', 'overall',                      'round', 'guid_x', 'grade_x', 'text', 'MM_overall'])
y = df[['MM_overall']]
 


# In[28]:


#create a train test split with clean wide data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123)

