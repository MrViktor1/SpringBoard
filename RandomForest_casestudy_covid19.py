#!/usr/bin/env python
# coding: utf-8

# ## Random Forest
# 
# Random Forest is an ensemble of Decision Trees. With a few exceptions, a `RandomForestClassifier` has all the hyperparameters of a `DecisionTreeClassifier` (to control how trees are grown), plus all the hyperparameters of a `BaggingClassifier` to control the ensemble itself.
# 
# The Random Forest algorithm introduces extra randomness when growing trees; instead of searching for the very best feature when splitting a node, it searches for the best feature among a random subset of features. This results in a greater tree diversity, which (once again) trades a higher bias for a lower variance, generally yielding an overall better model. The following `BaggingClassifier` is roughly equivalent to the previous `RandomForestClassifier`. Run the cell below to visualize a single estimator from a random forest model, using the Iris dataset to classify the data into the appropriate species.

# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(iris.data, iris.target)
# Extract single tree
estimator = model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# Notice how each split seperates the data into buckets of similar observations. This is a single tree and a relatively simple classification dataset, but the same method is used in a more complex dataset with greater depth to the trees.

# ## Coronavirus
# Coronavirus disease (COVID-19) is an infectious disease caused by a new virus.
# The disease causes respiratory illness (like the flu) with symptoms such as a cough, fever, and in more severe cases, difficulty breathing. You can protect yourself by washing your hands frequently, avoiding touching your face, and avoiding close contact (1 meter or 3 feet) with people who are unwell. An outbreak of COVID-19 started in December 2019 and at the time of the creation of this project was continuing to spread throughout the world. Many governments recommended only essential outings to public places and closed most business that do not serve food or sell essential items. An excellent [spatial dashboard](https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6) built by Johns Hopkins shows the daily confirmed cases by country. 
# 
# This case study was designed to drive home the important role that data science plays in real-world situations like this pandemic. This case study uses the Random Forest Classifier and a dataset from the South Korean cases of COVID-19 provided on [Kaggle](https://www.kaggle.com/kimjihoo/coronavirusdataset) to encourage research on this important topic. The goal of the case study is to build a Random Forest Classifier to predict the 'state' of the patient.

# First, please load the needed packages and modules into Python. Next, load the data into a pandas dataframe for ease of use.

# In[3]:


import os
import pandas as pd
from datetime import datetime,timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor


# In[4]:


pip install --upgrade pandas


# In[5]:


#url ='/kaggle/input/coronavirusdataset/PatientInfo.csv'
df = pd.read_csv('SouthKoreacoronavirusdataset-20200630T044816Z-001/SouthKoreacoronavirusdataset/PatientInfo.csv')
df.head()


# In[6]:


df.shape


# In[7]:


#Counts of null values 
na_df=pd.DataFrame(df.isnull().sum().sort_values(ascending=False)).reset_index()
na_df.columns = ['VarName', 'NullCount']
na_df[(na_df['NullCount']>0)]


# In[8]:


#counts of response variable values
df.state.value_counts()


#  **<font color='teal'> Create a new column named 'n_age' which is the calculated age based on the birth year column.</font>**

# In[9]:


df['n_age'] = 2023 - df['birth_year']


# In[10]:


df['n_age'].describe


# ### Handle Missing Values

#  **<font color='teal'> Print the number of missing values by column.</font>**

# In[11]:


df.isna().sum()


# In[12]:


df.info()


#  **<font color='teal'> Fill the 'disease' missing values with 0 and remap the True values to 1.</font>**

# In[13]:


df['disease'] = df['disease'].replace(np.nan, 0)


# In[14]:


df['disease'] = df['disease'].replace(True, 1)


#  **<font color='teal'> Fill null values in the following columns with their mean: 'global_number','birth_year','infection_order','infected_by'and 'contact_number'</font>**

# In[15]:


df[['global_num','birth_year','infection_order','infected_by','contact_number']] = df[['global_num','birth_year','infection_order','infected_by','contact_number']].fillna(df[['global_num','birth_year','infection_order','infected_by','contact_number']].mean())


#  **<font color='teal'> Fill the rest of the missing values with any method.</font>**

# In[16]:


df = df.fillna(0) 


#  **<font color='teal'> Check for any remaining null values.</font>**

# In[17]:


df.info()


# In[18]:


df.head()


# Remove date columns from the data.
# 

# In[19]:


df = df.drop(['symptom_onset_date','confirmed_date','released_date','deceased_date'],axis =1)


# Review the count of unique values by column.

# In[20]:


print(df.nunique())


# Review the percent of unique values by column.

# In[21]:


print(df.nunique()/df.shape[0])


# Review the range of values per column.

# In[22]:


df.describe().T


# ### Check for duplicated rows

# In[23]:


duplicateRowsDF = df[df.duplicated()]
duplicateRowsDF


# Print the categorical columns and their associated levels.

# In[24]:


dfo = df.select_dtypes(include=['object'], exclude=['datetime'])
dfo.shape
#get levels for all variables
vn = pd.DataFrame(dfo.nunique()).reset_index()
vn.columns = ['VarName', 'LevelsCount']
vn.sort_values(by='LevelsCount', ascending =False)
vn


# **<font color='teal'> Plot the correlation heat map for the features.</font>**

# In[25]:


sns.heatmap(df.corr(numeric_only = True))


# **<font color='teal'> Plot the boxplots to check for outliers. </font>**

# In[26]:


plt.figure(figsize=(30, 15))
df.boxplot()


# In[27]:



numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# plotting boxplots for each numerical column
for col in numerical_cols:
    plt.figure(figsize=(6, 8))
    sns.boxplot(data=df[col])
    plt.title(f"Boxplot for {col}")
plt.show()


# **<font color='teal'> Create dummy features for object type features. </font>**

# In[36]:


df_d.dtypes


# In[37]:


features = ["sex", "age", "country", "province", "city", "infection_case"]
df_dummies = pd.get_dummies(df[features], dtype=int)
df_d = pd.concat([df, df_dummies], axis = 1)
df_d = df_d.drop(features, axis = 1)
df_d.head()


# In[38]:


df_d = df_d.set_index("patient_id")


# ### Split the data into test and train subsamples

# In[45]:


# dont forget to define your X and y
state_mapping = {'isolated': 0, 'deceased': 1, 'released': 2}
df_d['state_encoded'] = df_d['state'].map(state_mapping)


# In[57]:


df_d_dna = df_d.dropna(subset = 'state_encoded')


# In[59]:


df_d_dna['state_encoded'].unique()


# In[60]:


from sklearn.model_selection import train_test_split



X = df_d_dna.drop(columns = ['state', 'state_encoded'])
y = df_d_dna['state_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)


# ### Scale data to prep for model creation

# In[61]:


#scale data
from sklearn import preprocessing
import numpy as np
# build scaler based on training data and apply it to test data to then also scale the test data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[62]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score
from sklearn.metrics import accuracy_score,log_loss
from matplotlib import pyplot


# ### Fit Random Forest Classifier
# The fit model shows an overall accuracy of 80% which is great and indicates our model was effectively able to identify the status of a patients in the South Korea dataset.

# In[63]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=300, random_state = 1,n_jobs=-1)
model_res = clf.fit(X_train_scaled, y_train)
y_pred = model_res.predict(X_test_scaled)
y_pred_prob = model_res.predict_proba(X_test_scaled)
lr_probs = y_pred_prob[:,1]
ac = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print('Random Forest: Accuracy=%.3f' % (ac))

print('Random Forest: f1-score=%.3f' % (f1))


# ### Create Confusion Matrix Plots
# Confusion matrices are great ways to review your model performance for a multi-class classification problem. Being able to identify which class the misclassified observations end up in is a great way to determine if you need to build additional features to improve your overall model. In the example below we plot a regular counts confusion matrix as well as a weighted percent confusion matrix. The percent confusion matrix is particulary helpful when you have unbalanced class sizes.

# In[64]:


class_names=['isolated','released','missing','deceased'] # name  of classes


# In[65]:


import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
#plt.savefig('figures/RF_cm_multi_class.png')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
#plt.savefig('figures/RF_cm_proportion_multi_class.png', bbox_inches="tight")
plt.show()


# ### Plot feature importances
# The random forest algorithm can be used as a regression or classification model. In either case it tends to be a bit of a black box, where understanding what's happening under the hood can be difficult. Plotting the feature importances is one way that you can gain a perspective on which features are driving the model predictions.

# In[66]:


feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())[:30]
sorted_idx = np.argsort(feature_importance)[:30]

pos = np.arange(sorted_idx.shape[0]) + .5
print(pos.size)
sorted_idx.size
plt.figure(figsize=(10,10))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# The popularity of random forest is primarily due to how well it performs in a multitude of data situations. It tends to handle highly correlated features well, where as a linear regression model would not. In this case study we demonstrate the performance ability even with only a few features and almost all of them being highly correlated with each other.
# Random Forest is also used as an efficient way to investigate the importance of a set of features with a large data set. Consider random forest to be one of your first choices when building a decision tree, especially for multiclass classifications.
