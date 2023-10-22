#!/usr/bin/env python
# coding: utf-8

# # **Springboard Decision Tree Specialty Coffee Case Study - Tier 3**
# 
# 
# 

# # The Scenario
# 
# Imagine you've just finished the Springboard Data Science Career Track course, and have been hired by a rising popular specialty coffee company - RR Diner Coffee - as a data scientist. Congratulations!
# 
# RR Diner Coffee sells two types of thing:
# - specialty coffee beans, in bulk (by the kilogram only) 
# - coffee equipment and merchandise (grinders, brewing equipment, mugs, books, t-shirts).
# 
# RR Diner Coffee has three stores, two in Europe and one in the USA. The flagshap store is in the USA, and everything is quality assessed there, before being shipped out. Customers further away from the USA flagship store have higher shipping charges. 
# 
# You've been taken on at RR Diner Coffee because the company are turning towards using data science and machine learning to systematically make decisions about which coffee farmers they should strike deals with. 
# 
# RR Diner Coffee typically buys coffee from farmers, processes it on site, brings it back to the USA, roasts it, packages it, markets it, and ships it (only in bulk, and after quality assurance) to customers internationally. These customers all own coffee shops in major cities like New York, Paris, London, Hong Kong, Tokyo, and Berlin. 
# 
# Now, RR Diner Coffee has a decision about whether to strike a deal with a legendary coffee farm (known as the **Hidden Farm**) in rural China: there are rumours their coffee tastes of lychee and dark chocolate, while also being as sweet as apple juice. 
# 
# It's a risky decision, as the deal will be expensive, and the coffee might not be bought by customers. The stakes are high: times are tough, stocks are low, farmers are reverting to old deals with the larger enterprises and the publicity of selling *Hidden Farm* coffee could save the RR Diner Coffee business. 
# 
# Your first job, then, is ***to build a decision tree to predict how many units of the Hidden Farm Chinese coffee will be purchased by RR Diner Coffee's most loyal customers.*** 
# 
# To this end, you and your team have conducted a survey of 710 of the most loyal RR Diner Coffee customers, collecting data on the customers':
# - age
# - gender 
# - salary 
# - whether they have bought at least one RR Diner Coffee product online
# - their distance from the flagship store in the USA (standardized to a number between 0 and 11) 
# - how much they spent on RR Diner Coffee products on the week of the survey 
# - how much they spent on RR Diner Coffee products in the month preeding the survey
# - the number of RR Diner coffee bean shipments each customer has ordered over the preceding year. 
# 
# You also asked each customer participating in the survey whether they would buy the Hidden Farm coffee, and some (but not all) of the customers gave responses to that question. 
# 
# You sit back and think: if more than 70% of the interviewed customers are likely to buy the Hidden Farm coffee, you will strike the deal with the local Hidden Farm farmers and sell the coffee. Otherwise, you won't strike the deal and the Hidden Farm coffee will remain in legends only. There's some doubt in your mind about whether 70% is a reasonable threshold, but it'll do for the moment. 
# 
# To solve the problem, then, you will build a decision tree to implement a classification solution. 
# 
# 
# -------------------------------
# As ever, this notebook is **tiered**, meaning you can elect that tier that is right for your confidence and skill level. There are 3 tiers, with tier 1 being the easiest and tier 3 being the hardest. This is ***tier 3***, so it will be challenging. 
# 
# **1. Sourcing and loading** 
# - Import packages
# - Load data
# - Explore the data
# 
#  
# **2. Cleaning, transforming and visualizing**
# - Cleaning the data
# - Train/test split
#   
#   
# **3. Modelling** 
# - Model 1: Entropy model - no max_depth
# - Model 2: Gini impurity model - no max_depth
# - Model 3: Entropy model - max depth 3
# - Model 4: Gini impurity model - max depth 3
# 
# 
# **4. Evaluating and concluding** 
# - How many customers will buy Hidden Farm coffee?
# - Decision
# 
# **5. Random Forest** 
# - Import necessary modules
# - Model
# - Revise conclusion
#     

# # 0. Overview
# 
# This notebook uses decision trees to determine whether the factors of salary, gender, age, how much money the customer spent last week and during the preceding month on RR Diner Coffee products, how many kilogram coffee bags the customer bought over the last year, whether they have bought at least one RR Diner Coffee product online, and their distance from the flagship store in the USA, could predict whether customers would purchase the Hidden Farm coffee if a deal with its farmers were struck. 

# # 1. Sourcing and loading
# ## 1a. Import Packages

# In[32]:



conda install graphviz


# In[1]:


import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO  
from IPython.display import Image  
import pydotplus


# ## 1b. Load data 

# In[2]:


# Read in the data to a variable called coffeeData
coffeeData = pd.read_csv("Data/RRDinerCoffeeData.csv")


# ## 1c. Explore the data

# As we've seen, exploration entails doing things like checking out the **initial appearance** of the data with head(), the **dimensions** of our data with .shape, the **data types** of the variables with .info(), the **number of non-null values**, how much **memory** is being used to store the data, and finally the major summary statistcs capturing **central tendancy, dispersion and the null-excluding shape of the dataset's distribution**. 
# 
# How much of this can you do yourself by this point in the course? Have a real go. 

# In[3]:


# Call head() on your data 
coffeeData.head()


# In[4]:


# Call .shape on your data
coffeeData.shape


# In[5]:


# Call info() on your data
coffeeData.info()


# In[6]:


# Call describe() on your data to get the relevant summary statistics for your data 
coffeeData.describe()


# # 2. Cleaning, transforming and visualizing
# ## 2a. Cleaning the data

# Some datasets don't require any cleaning, but almost all do. This one does. We need to replace '1.0' and '0.0' in the 'Decision' column by 'YES' and 'NO' respectively, clean up the values of the 'gender' column, and change the column names to words which maximize meaning and clarity. 

# First, let's change the name of `spent_week`, `spent_month`, and `SlrAY` to `spent_last_week` and `spent_last_month` and `salary` respectively.

# In[7]:


# Check out the names of our data's columns 
coffeeData.columns


# In[8]:


# Make the relevant name changes to spent_week and spent_per_week.
coffeeData.rename(columns = {'spent_week':'spent_last_week', 'spent_month':'spent_last_month','SlrAY':'salary'}, 
                  inplace=True)


# In[9]:


# Check out the column names
coffeeData.columns


# In[10]:


# Let's have a closer look at the gender column. Its values need cleaning.
coffeeData['Gender'].describe()


# In[11]:


# See the gender column's unique values 
coffeeData['Gender'].unique()


# We can see a bunch of inconsistency here.
# 
# Use replace() to make the values of the `gender` column just `Female` and `Male`.

# In[12]:


# Replace all alternate values for the Female entry with 'Female'
coffeeData['Gender'].replace(to_replace=['female','f','F', 'FEMALE', 'f '], value = 'Female', inplace=True)


# In[13]:


# Check out the unique values for the 'gender' column
coffeeData['Gender'].unique()


# In[14]:


# Replace all alternate values with "Male"
coffeeData['Gender'].replace(to_replace=['MALE', 'M', 'male'], value = 'Male', inplace=True)


# In[15]:


# Let's check the unique values of the column "gender"
coffeeData['Gender'].unique()


# In[16]:


# Check out the unique values of the column 'Decision'
coffeeData['Decision'].unique()


# We now want to replace `1.0` and `0.0` in the `Decision` column by `YES` and `NO` respectively.

# In[43]:


# Replace 1.0 and 0.0 by 'Yes' and 'No'
coffeeData['Decision'].replace(1.0, 'YES', inplace=True)
coffeeData['Decision'].replace(0.0,'NO', inplace=True)


# In[44]:


# Check that our replacing those values with 'YES' and 'NO' worked, with unique()
coffeeData['Decision'].unique()


# ## 2b. Train/test split
# To execute the train/test split properly, we need to do five things: 
# 1. Drop all rows with a null value in the `Decision` column, and save the result as NOPrediction: a dataset that will contain all known values for the decision 
# 2. Visualize the data using scatter and boxplots of several variables in the y-axis and the decision on the x-axis
# 3. Get the subset of coffeeData with null values in the `Decision` column, and save that subset as Prediction
# 4. Divide the NOPrediction subset into X and y, and then further divide those subsets into train and test subsets for X and y respectively
# 5. Create dummy variables to deal with categorical inputs

# ### 1. Drop all null values within the `Decision` column, and save the result as NoPrediction

# In[45]:


# NoPrediction will contain all known values for the decision
# Call dropna() on coffeeData, and store the result in a variable NOPrediction 
# Call describe() on the Decision column of NoPrediction after calling dropna() on coffeeData
NoPrediction = coffeeData.dropna()
NoPrediction['Decision'].describe()


# ### 2. Visualize the data using scatter and boxplots of several variables in the y-axis and the decision on the x-axis

# In[46]:


# Exploring our new NOPrediction dataset
# Make a boxplot on NOPrediction where the x axis is Decision, and the y axis is spent_last_week
sns.boxplot(x='Decision', y='spent_last_week', data=NoPrediction)


# Can you admissibly conclude anything from this boxplot? Write your answer here:
# the 25th percentile of those who answered NO have not bought coffee in the last week. the mean amount spent last week for those who said YES is higher then the mean amount from those who responded NO
# 

# In[47]:


NoPrediction['Distance'].describe()


# In[48]:


# Make a scatterplot on NOPrediction, where x is distance, y is spent_last_month and hue is Decision 
sns.scatterplot(y="spent_last_month", x= "Distance", hue = "Decision", data =NoPrediction)


# Can you admissibly conclude anything from this scatterplot? Remember: we are trying to build a tree to classify unseen examples. Write your answer here:
# The larger the distance the more likely the answer was NO. The more spent the last month the more likely the answer was YES. When ploted together it seems to provide a good split for costumers that will answer YES vs NO.

# ### 3. Get the subset of coffeeData with null values in the Decision column, and save that subset as Prediction

# In[49]:


# Get just those rows whose value for the Decision column is null  
Prediction = coffeeData[coffeeData['Decision'].isna()] 


# In[50]:


# Call describe() on Prediction
Prediction.describe()


# ### 4. Divide the NOPrediction subset into X and y

# In[51]:


# Check the names of the columns of NOPrediction
NoPrediction.columns


# In[52]:


# Let's do our feature selection.
# Make a variable called 'features', and a list containing the strings of every column except "Decision"
features = ["Age", "Gender", "num_coffeeBags_per_year", "spent_last_week", "spent_last_month",
       "salary", "Distance", "Online"]
# Make an explanatory variable called X, and assign it: NoPrediction[features]
X = NoPrediction[features]
# Make a dependent variable called y, and assign it: NoPrediction.Decision
y = NoPrediction.Decision


# ### 5. Create dummy variables to deal with categorical inputs
# One-hot encoding replaces each unique value of a given column with a new column, and puts a 1 in the new column for a given row just if its initial value for the original column matches the new column. Check out [this resource](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) if you haven't seen one-hot-encoding before. 
# 
# **Note**: We will do this before we do our train/test split as to do it after could mean that some categories only end up in the train or test split of our data by chance and this would then lead to different shapes of data for our `X_train` and `X_test` which could/would cause downstream issues when fitting or predicting using a trained model.

# In[53]:


# One-hot encode all features in X.
X = pd.get_dummies(X)


# ### 6. Further divide those subsets into train and test subsets for X and y respectively: X_train, X_test, y_train, y_test

# In[54]:


# Call train_test_split on X, y. Make the test_size = 0.25, and random_state = 246
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.25, random_state = 246)


# # 3. Modelling
# It's useful to look at the scikit-learn documentation on decision trees https://scikit-learn.org/stable/modules/tree.html before launching into applying them. If you haven't seen them before, take a look at that link, in particular the section `1.10.5.` 

# ## Model 1: Entropy model - no max_depth
# 
# We'll give you a little more guidance here, as the Python is hard to deduce, and scikitlearn takes some getting used to.
# 
# Theoretically, let's remind ourselves of what's going on with a decision tree implementing an entropy model.
# 
# Ross Quinlan's **ID3 Algorithm** was one of the first, and one of the most basic, to use entropy as a metric.
# 
# **Entropy** is a measure of how uncertain we are about which category the data-points fall into at a given point in the tree. The **Information gain** of a specific feature with a threshold (such as 'spent_last_month <= 138.0') is the difference in entropy that exists before and after splitting on that feature; i.e., the information we gain about the categories of the data-points by splitting on that feature and that threshold. 
# 
# Naturally, we want to minimize entropy and maximize information gain. Quinlan's ID3 algorithm is designed to output a tree such that the features at each node, starting from the root, and going all the way down to the leaves, have maximial information gain. We want a tree whose leaves have elements that are *homogeneous*, that is, all of the same category. 
# 
# The first model will be the hardest. Persevere and you'll reap the rewards: you can use almost exactly the same code for the other models. 

# In[55]:


# Declare a variable called entr_model and use tree.DecisionTreeClassifier. 
entr_model = tree.DecisionTreeClassifier(criterion="entropy", random_state=5)
# Call fit() on entr_model
entr_model.fit(X_train, y_train)
# Call predict() on entr_model with X_test passed to it, and assign the result to a variable y_pred 
y_pred = entr_model.predict(X_test)
# Call Series on our y_pred variable with the following: pd.Series(y_pred)
pd.Series(y_pred)
# Check out entr_model
entr_model


# In[56]:


# Now we want to visualize the tree
dot_data = StringIO()

# We can do so with export_graphviz
tree.export_graphviz(entr_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=X_train.columns,class_names = ["NO", "YES"]) 

# Alternatively for class_names use entr_model.classes_
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ## Model 1: Entropy model - no max_depth: Interpretation and evaluation

# In[57]:


# Run this block for model evaluation metrics 
print("Model Entropy - no max depth")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score for "Yes"' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Precision score for "No"' , metrics.precision_score(y_test,y_pred, pos_label = "NO"))
print('Recall score for "Yes"' , metrics.recall_score(y_test,y_pred, pos_label = "YES"))
print('Recall score for "No"' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))


# What can you infer from these results? Write your conclusions here:
# This model fits very to the test data. We showed 99% accuracy and only misslabled a few of the costumers that would answer Yes. 

# ## Model 2: Gini impurity model - no max_depth
# 
# Gini impurity, like entropy, is a measure of how well a given feature (and threshold) splits the data into categories.
# 
# Their equations are similar, but Gini impurity doesn't require logorathmic functions, which can be computationally expensive. 

# In[62]:


# Make a variable called gini_model, and assign it exactly what you assigned entr_model with above, but with the
# criterion changed to 'gini'
gini_model = tree.DecisionTreeClassifier(criterion="gini", random_state=5)

# Call fit() on the gini_model as you did with the entr_model
gini_model.fit(X_train, y_train)

# Call predict() on the gini_model as you did with the entr_model 
y_pred = gini_model.predict(X_test)
# Turn y_pred into a series, as before
pd.Series(g_y_pred)

# Check out gini_model
gini_model


# In[63]:


# As before, but make the model name gini_model
dot_data = StringIO()
tree.export_graphviz(gini_model , out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=X_train.columns,class_names = ["NO", "YES"])

# Alternatively for class_names use gini_model.classes_
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[64]:


# Run this block for model evaluation
print("Model Gini impurity model")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))


# How do the results here compare to the previous model? Write your judgements here: 
# The accuracy and balanced accuracy both went down. 

# ## Model 3: Entropy model - max depth 3
# We're going to try to limit the depth of our decision tree, using entropy first.  
# 
# As you know, we need to strike a balance with tree depth. 
# 
# Insufficiently deep, and we're not giving the tree the opportunity to spot the right patterns in the training data.
# 
# Excessively deep, and we're probably going to make a tree that overfits to the training data, at the cost of very high error on the (hitherto unseen) test data. 
# 
# Sophisticated data scientists use methods like random search with cross-validation to systematically find a good depth for their tree. We'll start with picking 3, and see how that goes. 

# In[65]:


# Made a model as before, but call it entr_model2, and make the max_depth parameter equal to 3. 
# Execute the fitting, predicting, and Series operations as before
entr_model2 = tree.DecisionTreeClassifier(criterion="entropy", random_state=5, max_depth=3)

entr_model2.fit(X_train, y_train)
# Call predict() on entr_model with X_test passed to it, and assign the result to a variable y_pred 
y_pred = entr_model2.predict(X_test)
# Call Series on our y_pred variable with the following: pd.Series(y_pred)
pd.Series(y_pred)
# Check out entr_model
entr_model2


# In[66]:


# As before, we need to visualize the tree to grasp its nature
dot_data = StringIO()
tree.export_graphviz(entr_model2 , out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=X_train.columns,class_names = ["NO", "YES"])

# Alternatively for class_names use 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[67]:


# Run this block for model evaluation 
print("Model Entropy model max depth 3")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score for "Yes"' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score for "No"' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))


# So our accuracy decreased, but is this certainly an inferior tree to the max depth original tree we did with Model 1? Write your conclusions here: 
# It performed worsed on our test set which is a bad sign but it may be less prone to be overfit to the data that came from out surveyers and could preform better on the general costumer population. 

# ## Model 4: Gini impurity  model - max depth 3
# We're now going to try the same with the Gini impurity model. 

# In[68]:


# As before, make a variable, but call it gini_model2, and ensure the max_depth parameter is set to 3
gini_model2 = tree.DecisionTreeClassifier(criterion="gini", random_state=5, max_depth=3)

# Do the fit, predict, and series transformations as before. 
gini_model2.fit(X_train, y_train)
# Call predict() on entr_model with X_test passed to it, and assign the result to a variable y_pred 
y_pred = gini_model2.predict(X_test)
# Call Series on our y_pred variable with the following: pd.Series(y_pred)
pd.Series(y_pred)
# Check out entr_model
gini_model2


# In[69]:


dot_data = StringIO()
tree.export_graphviz(gini_model2 , out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=X_train.columns,class_names = ["NO", "YES"])

# Alternatively for class_names use 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[70]:


print("Gini impurity  model - max depth 3")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))


# Now this is an elegant tree. Its accuracy might not be the highest, but it's still the best model we've produced so far. Why is that? Write your answer here: 
# 
# It doesn't overfit the training data and it scored well on the test data.

# # 4. Evaluating and concluding
# ## 4a. How many customers will buy Hidden Farm coffee? 
# Let's first ascertain how many loyal customers claimed, in the survey, that they will purchase the Hidden Farm coffee. 

# In[72]:


# Call value_counts() on the 'Decision' column of the original coffeeData
coffeeData['Decision'].value_counts()


# Let's now determine the number of people that, according to the model, will be willing to buy the Hidden Farm coffee. 
# 1. First we subset the Prediction dataset into `new_X` considering all the variables except `Decision` 
# 2. Use that dataset to predict a new variable called `potential_buyers`

# In[74]:


# Feature selection
# Make a variable called feature_cols, and assign it a list containing all the column names except 'Decision'
feature_cols = ["Age", "Gender", "num_coffeeBags_per_year", "spent_last_week", "spent_last_month",
       "salary", "Distance", "Online"]
# Make a variable called new_X, and assign it the subset of Prediction, containing just the feature_cols 
new_X = Prediction[feature_cols]


# In[80]:


# Call get_dummies() on the Pandas object pd, with new_X plugged in, to one-hot encode all features in the training set
new_X = pd.get_dummies(new_X)
# Make a variable called potential_buyers, and assign it the result of calling predict() on a model of your choice; 
# don't forget to pass new_X to predict()
potential_buyers = gini_model2.predict(new_X)


# In[82]:


# Let's get the numbers of YES's and NO's in the potential buyers 
# Call unique() on np, and pass potential_buyers and return_counts=True 
np.unique(potential_buyers, return_counts=True)


# The total number of potential buyers is 303 + 183 = 486

# In[84]:


# Print the total number of surveyed people 
coffeeData['Gender'].count()


# In[85]:


# Let's calculate the proportion of buyers
486/702


# In[88]:


# Print the percentage of people who want to buy the Hidden Farm coffee, by our model 
print((486/702)*100)


# ## 4b. Decision
# Remember how you thought at the start: if more than 70% of the interviewed customers are likely to buy the Hidden Farm coffee, you will strike the deal with the local Hidden Farm farmers and sell the coffee. Otherwise, you won't strike the deal and the Hidden Farm coffee will remain in legends only. Well now's crunch time. Are you going to go ahead with that idea? If so, you won't be striking the deal with the Chinese farmers. 
# 
# They're called `decision trees`, aren't they? So where's the decision? What should you do? (Cue existential cat emoji). 
# 
# Ultimately, though, we can't write an algorithm to actually *make the business decision* for us. This is because such decisions depend on our values, what risks we are willing to take, the stakes of our decisions, and how important it us for us to *know* that we will succeed. What are you going to do with the models you've made? Are you going to risk everything, strike the deal with the *Hidden Farm* farmers, and sell the coffee? 
# 
# The philosopher of language Jason Stanley once wrote that the number of doubts our evidence has to rule out in order for us to know a given proposition depends on our stakes: the higher our stakes, the more doubts our evidence has to rule out, and therefore the harder it is for us to know things. We can end up paralyzed in predicaments; sometimes, we can act to better our situation only if we already know certain things, which we can only if our stakes were lower and we'd *already* bettered our situation. 
# 
# Data science and machine learning can't solve such problems. But what it can do is help us make great use of our data to help *inform* our decisions.

# ## 5. Random Forest
# You might have noticed an important fact about decision trees. Each time we run a given decision tree algorithm to make a prediction (such as whether customers will buy the Hidden Farm coffee) we will actually get a slightly different result. This might seem weird, but it has a simple explanation: machine learning algorithms are by definition ***stochastic***, in that their output is at least partly determined by randomness. 
# 
# To account for this variability and ensure that we get the most accurate prediction, we might want to actually make lots of decision trees, and get a value that captures the centre or average of the outputs of those trees. Luckily, there's a method for this, known as the ***Random Forest***. 
# 
# Essentially, Random Forest involves making lots of trees with similar properties, and then performing summary statistics on the outputs of those trees to reach that central value. Random forests are hugely powerful classifers, and they can improve predictive accuracy and control over-fitting. 
# 
# Why not try to inform your decision with random forest? You'll need to make use of the RandomForestClassifier function within the sklearn.ensemble module, found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 

# ### 5a. Import necessary modules

# In[89]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# ### 5b. Model
# You'll use your X_train and y_train variables just as before.
# 
# You'll then need to make a variable (call it firstRFModel) to store your new Random Forest model. You'll assign this variable the result of calling RandomForestClassifier().
# 
# Then, just as before, you'll call fit() on that firstRFModel variable, and plug in X_train and y_train.
# 
# Finally, you should make a variable called y_pred, and assign it the result of calling the predict() method on your new firstRFModel, with the X_test data passed to it. 

# In[91]:


# Plug in appropriate max_depth and random_state parameters 
firstRFModel = RandomForestClassifier(max_depth = 3, random_state=5)
# Model and fit
firstRFModel.fit(X_train, y_train)

y_pred = firstRFModel.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))


# ### 5c. Revise conclusion
# 
# Has your conclusion changed? Or is the result of executing random forest the same as your best model reached by a single decision tree? 
# The random forest model scored lower on the decision tree matrix. This is a more conservative model that averages many models and it appears to underfit the data.
