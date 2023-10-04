#!/usr/bin/env python
# coding: utf-8

# # Frequentist Inference Case Study - Part B

# ## Learning objectives

# Welcome to Part B of the Frequentist inference case study! The purpose of this case study is to help you apply the concepts associated with Frequentist inference in Python. In particular, you'll practice writing Python code to apply the following statistical concepts: 
# * the _z_-statistic
# * the _t_-statistic
# * the difference and relationship between the two
# * the Central Limit Theorem, including its assumptions and consequences
# * how to estimate the population mean and standard deviation from a sample
# * the concept of a sampling distribution of a test statistic, particularly for the mean
# * how to combine these concepts to calculate a confidence interval

# In the previous notebook, we used only data from a known normal distribution. **You'll now tackle real data, rather than simulated data, and answer some relevant real-world business problems using the data.**

# ## Hospital medical charges

# Imagine that a hospital has hired you as their data scientist. An administrator is working on the hospital's business operations plan and needs you to help them answer some business questions. 
# 
# In this assignment notebook, you're going to use frequentist statistical inference on a data sample to answer the questions:
# * has the hospital's revenue stream fallen below a key threshold?
# * are patients with insurance really charged different amounts than those without?
# 
# Answering that last question with a frequentist approach makes some assumptions, and requires some knowledge, about the two groups.

# We are going to use some data on medical charges obtained from [Kaggle](https://www.kaggle.com/easonlai/sample-insurance-claim-prediction-dataset). 
# 
# For the purposes of this exercise, assume the observations are the result of random sampling from our single hospital. Recall that in the previous assignment, we introduced the Central Limit Theorem (CLT), and its consequence that the distributions of sample statistics approach a normal distribution as $n$ increases. The amazing thing about this is that it applies to the sampling distributions of statistics that have been calculated from even highly non-normal distributions of data! Recall, also, that hypothesis testing is very much based on making inferences about such sample statistics. You're going to rely heavily on the CLT to apply frequentist (parametric) tests to answer the questions in this notebook.

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import stats
from numpy.random import seed
medical = pd.read_csv('insurance2.csv')


# In[4]:


medical.shape


# In[5]:


medical.head()


# __Q1:__ Plot the histogram of charges and calculate the mean and standard deviation. Comment on the appropriateness of these statistics for the data.

# __A:__ The mean seems plausible but the standard deviation seems large.

# In[2]:


charges = medical['charges']


# In[3]:


_ = plt.hist(charges)
_ = plt.xlabel('charges')
_ = plt.ylabel('number of occurances')
_ = plt.title('Hospital Charges Per Patient')


# In[4]:


print(np.mean(charges))
print(np.std(charges, ddof=1))


# __Q2:__ The administrator is concerned that the actual average charge has fallen below 12,000, threatening the hospital's operational model. On the assumption that these data represent a random sample of charges, how would you justify that these data allow you to answer that question? And what would be the most appropriate frequentist test, of the ones discussed so far, to apply?

# __A:__  I would say that given the large sample size using a 95 confidence interval plus or minus the standard error would give us a range that we are 95% confident that the true actual average charge lies.  

# __Q3:__ Given the nature of the administrator's concern, what is the appropriate confidence interval in this case? A ***one-sided*** or ***two-sided*** interval? (Refresh your understanding of this concept on p. 399 of the *AoS*). Calculate the critical value and the relevant 95% confidence interval for the mean, and comment on whether the administrator should be concerned.

# __A:__ 1 sided interval would be best as the only tail concerning us is if the true mean is less than 12,000 so any sample with a mean 2 standard deviations higher than the true mean would not be problematic.
# The 95% confidence interval is 13270 plus or minus 19974
# because the true mean could be much larger or smaller than the sample we can't make a comment on if the administrator should be worried. 

# In[20]:


np.std(charges) * 1.65


# In[22]:





# In[31]:


len(uninsured)


# The administrator then wants to know whether people with insurance really are charged a different amount to those without.
# 
# __Q4:__ State the null and alternative hypothesis here. Use the _t_-test for the difference between means, where the pooled standard deviation of the two groups is given by:
# \begin{equation}
# s_p = \sqrt{\frac{(n_0 - 1)s^2_0 + (n_1 - 1)s^2_1}{n_0 + n_1 - 2}}
# \end{equation}
# 
# and the *t*-test statistic is then given by:
# 
# \begin{equation}
# t = \frac{\bar{x}_0 - \bar{x}_1}{s_p \sqrt{1/n_0 + 1/n_1}}.
# \end{equation}
# 
# (If you need some reminding of the general definition of ***t-statistic***, check out the definition on p. 404 of *AoS*). 
# 
# What assumption about the variances of the two groups are we making here?

# __A:__ 
# Ho: There is no statistically significant difference between the charges between patients with insurance and patients without insurance. 
# Ha: There is a statistically significant difference between the charges between patients with insurance and patients without insurance. 
# By using a pooled standard deviation we are assuming the two groups have a similar varriance. 

# __Q5:__ Perform this hypothesis test both manually, using the above formulae, and then using the appropriate function from [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests) (hint, you're looking for a function to perform a _t_-test on two independent samples). For the manual approach, calculate the value of the test statistic and then its probability (the p-value). Verify you get the same results from both.

# __A:__ The result of the t-statistic is 11, well above 2 standard deviations from the mean meaning the p value would be much less than .05 indicating there is a statistical difference between the the insured and uninsured groups. I would accept the alternative hypothesis in this instance. 
# 

# In[19]:


insured = medical.loc[medical['insuranceclaim'] == 1, 'charges']


# In[21]:


uninsured = medical.loc[medical['insuranceclaim'] == 0, 'charges']


# In[33]:


Sp = np.sqrt(((len(uninsured)-1)*np.std(uninsured)**2 + (len(insured)-1)*np.std(insured)**2)/(len(uninsured)+len(insured)-2))


# In[37]:


ts = (np.mean(uninsured) - np.mean(insured))/(Sp*np.sqrt(1/len(uninsured)+1/len(insured)))
ts


# In[58]:


stats.ttest_ind(insured, uninsured, random_state = 47)


# Congratulations! Hopefully you got the exact same numerical results. This shows that you correctly calculated the numbers by hand. Secondly, you used the correct function and saw that it's much easier to use. All you need to do is pass your data to it.

# __Q6:__ Conceptual question: look through the documentation for statistical test functions in scipy.stats. You'll see the above _t_-test for a sample, but can you see an equivalent one for performing a *z*-test from a sample? Comment on your answer.

# __A:__ scipy.stats.zscore

# ## Learning outcomes

# Having completed this project notebook, you now have good hands-on experience:
# * using the central limit theorem to help you apply frequentist techniques to answer questions that pertain to very non-normally distributed data from the real world
# * performing inference using such data to answer business questions
# * forming a hypothesis and framing the null and alternative hypotheses
# * testing this using a _t_-test
