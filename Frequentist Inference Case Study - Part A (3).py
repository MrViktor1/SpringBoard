#!/usr/bin/env python
# coding: utf-8

# # Frequentist Inference Case Study - Part A 

# ## 1. Learning objectives

# Welcome to part A of the Frequentist inference case study! The purpose of this case study is to help you apply the concepts associated with Frequentist inference in Python. Frequentist inference is the process of deriving conclusions about an underlying distribution via the observation of data. In particular, you'll practice writing Python code to apply the following statistical concepts: 
# * the _z_-statistic
# * the _t_-statistic
# * the difference and relationship between the two
# * the Central Limit Theorem, including its assumptions and consequences
# * how to estimate the population mean and standard deviation from a sample
# * the concept of a sampling distribution of a test statistic, particularly for the mean
# * how to combine these concepts to calculate a confidence interval

# ## Prerequisites

# To be able to complete this notebook, you are expected to have a basic understanding of:
# * what a random variable is (p.400 of Professor Spiegelhalter's *The Art of Statistics, hereinafter AoS*)
# * what a population, and a population distribution, are (p. 397 of *AoS*)
# * a high-level sense of what the normal distribution is (p. 394 of *AoS*)
# * what the t-statistic is (p. 275 of *AoS*)
# 
# Happily, these should all be concepts with which you are reasonably familiar after having read ten chapters of Professor Spiegelhalter's book, *The Art of Statistics*.
# 
# We'll try to relate the concepts in this case study back to page numbers in *The Art of Statistics* so that you can focus on the Python aspects of this case study. The second part (part B) of this case study will involve another, more real-world application of these tools. 

# For this notebook, we will use data sampled from a known normal distribution. This allows us to compare our results with theoretical expectations.

# ## 2. An introduction to sampling from the normal distribution

# First, let's explore the ways we can generate the normal distribution. While there's a fair amount of interest in [sklearn](https://scikit-learn.org/stable/) within the machine learning community, you're likely to have heard of [scipy](https://docs.scipy.org/doc/scipy-0.15.1/reference/index.html) if you're coming from the sciences. For this assignment, you'll use [scipy.stats](https://docs.scipy.org/doc/scipy-0.15.1/reference/tutorial/stats.html) to complete your work. 
# 
# This assignment will require some digging around and getting your hands dirty (your learning is maximized that way)! You should have the research skills and the tenacity to do these tasks independently, but if you struggle, reach out to your immediate community and your mentor for help. 

# In[3]:


from scipy.stats import norm
from scipy.stats import t
import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt


# __Q1:__ Call up the documentation for the `norm` function imported above. (Hint: that documentation is [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)). What is the second listed method?

# In[35]:


help(norm)


# __A: pdf (probability density funtion)

# __Q2:__ Use the method that generates random variates to draw five samples from the standard normal distribution. 

# __A:__

# In[7]:


seed(47)
# draw five samples here
sample_five = norm.rvs(size=5)


# In[8]:


sample_five.mean()


# __Q3:__ What is the mean of this sample? Is it exactly equal to the value you expected? Hint: the sample was drawn from the standard normal distribution. If you want a reminder of the properties of this distribution, check out p. 85 of *AoS*. 

# __A:__ The expected mean is 0 with a standard deviation of +- 1 so a mean of 0.19 is acceptable for the mean of a standard normal distribution

# In[41]:


# Calculate and print the mean here, hint: use np.mean()
np.mean(sample_five)


# __Q4:__ What is the standard deviation of these numbers? Calculate this manually here as $\sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n}}$ (This is just the definition of **standard deviation** given by Professor Spiegelhalter on p.403 of *AoS*). Hint: np.sqrt() and np.sum() will be useful here and remember that numPy supports [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

# __A:__

# In[19]:


varriance = np.sum((sample_five - np.mean(sample_five))**2)/len(sample_five)


# In[20]:


std_dev = np.sqrt(varriance)
print(std_dev)


# In[21]:


sample_five.std()


# Here we have calculated the actual standard deviation of a small data set (of size 5). But in this case, this small data set is actually a sample from our larger (infinite) population. In this case, the population is infinite because we could keep drawing our normal random variates until our computers die! 
# 
# In general, the sample mean we calculate will not be equal to the population mean (as we saw above). A consequence of this is that the sum of squares of the deviations from the _population_ mean will be bigger than the sum of squares of the deviations from the _sample_ mean. In other words, the sum of squares of the deviations from the _sample_ mean is too small to give an unbiased estimate of the _population_ variance. An example of this effect is given [here](https://en.wikipedia.org/wiki/Bessel%27s_correction#Source_of_bias). Scaling our estimate of the variance by the factor $n/(n-1)$ gives an unbiased estimator of the population variance. This factor is known as [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). The consequence of this is that the $n$ in the denominator is replaced by $n-1$.
# 
# You can see Bessel's correction reflected in Professor Spiegelhalter's definition of **variance** on p. 405 of *AoS*.
# 
# __Q5:__ If all we had to go on was our five samples, what would be our best estimate of the population standard deviation? Use Bessel's correction ($n-1$ in the denominator), thus $\sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n-1}}$.

# __A:__

# In[22]:


std_dev_Bessel = np.sqrt(np.sum((sample_five - np.mean(sample_five))**2)/(len(sample_five)-1))


# In[23]:


std_dev_Bessel


# __Q6:__ Now use numpy's std function to calculate the standard deviation of our random samples. Which of the above standard deviations did it return?

# __A:__ It returned the regular standard deviation. 

# In[27]:


sample_five.std(ddof=1)


# __Q7:__ Consult the documentation for np.std() to see how to apply the correction for estimating the population parameter and verify this produces the expected result.

# __A:__ Adjusting the delta degrees of freedom will give the desired result of n - 1 

# In[ ]:





# In[ ]:





# ### Summary of section

# In this section, you've been introduced to the scipy.stats package and used it to draw a small sample from the standard normal distribution. You've calculated the average (the mean) of this sample and seen that this is not exactly equal to the expected population parameter (which we know because we're generating the random variates from a specific, known distribution). You've been introduced to two ways of calculating the standard deviation; one uses $n$ in the denominator and the other uses $n-1$ (Bessel's correction). You've also seen which of these calculations np.std() performs by default and how to get it to generate the other.

# You use $n$ as the denominator if you want to calculate the standard deviation of a sequence of numbers. You use $n-1$ if you are using this sequence of numbers to estimate the population parameter. This brings us to some terminology that can be a little confusing.
# 
# The population parameter is traditionally written as $\sigma$ and the sample statistic as $s$. Rather unhelpfully, $s$ is also called the sample standard deviation (using $n-1$) whereas the standard deviation of the sample uses $n$. That's right, we have the sample standard deviation and the standard deviation of the sample and they're not the same thing!
# 
# The sample standard deviation
# \begin{equation}
# s = \sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n-1}} \approx \sigma,
# \end{equation}
# is our best (unbiased) estimate of the population parameter ($\sigma$).
# 
# If your dataset _is_ your entire population, you simply want to calculate the population parameter, $\sigma$, via
# \begin{equation}
# \sigma = \sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n}}
# \end{equation}
# as you have complete, full knowledge of your population. In other words, your sample _is_ your population. It's worth noting that we're dealing with what Professor Spiegehalter describes on p. 92 of *AoS* as a **metaphorical population**: we have all the data, and we act as if the data-point is taken from a population at random. We can think of this population as an imaginary space of possibilities. 
# 
# If, however, you have sampled _from_ your population, you only have partial knowledge of the state of your population. In this case, the standard deviation of your sample is not an unbiased estimate of the standard deviation of the population, in which case you seek to estimate that population parameter via the sample standard deviation, which uses the $n-1$ denominator.

# Great work so far! Now let's dive deeper.

# ## 3. Sampling distributions

# So far we've been dealing with the concept of taking a sample from a population to infer the population parameters. One statistic we calculated for a sample was the mean. As our samples will be expected to vary from one draw to another, so will our sample statistics. If we were to perform repeat draws of size $n$ and calculate the mean of each, we would expect to obtain a distribution of values. This is the sampling distribution of the mean. **The Central Limit Theorem (CLT)** tells us that such a distribution will approach a normal distribution as $n$ increases (the intuitions behind the CLT are covered in full on p. 236 of *AoS*). For the sampling distribution of the mean, the standard deviation of this distribution is given by
# 
# \begin{equation}
# \sigma_{mean} = \frac{\sigma}{\sqrt n}
# \end{equation}
# 
# where $\sigma_{mean}$ is the standard deviation of the sampling distribution of the mean and $\sigma$ is the standard deviation of the population (the population parameter).

# This is important because typically we are dealing with samples from populations and all we know about the population is what we see in the sample. From this sample, we want to make inferences about the population. We may do this, for example, by looking at the histogram of the values and by calculating the mean and standard deviation (as estimates of the population parameters), and so we are intrinsically interested in how these quantities vary across samples. 
# 
# In other words, now that we've taken one sample of size $n$ and made some claims about the general population, what if we were to take another sample of size $n$? Would we get the same result? Would we make the same claims about the general population? This brings us to a fundamental question: _when we make some inference about a population based on our sample, how confident can we be that we've got it 'right'?_
# 
# We need to think about **estimates and confidence intervals**: those concepts covered in Chapter 7, p. 189, of *AoS*.

# Now, the standard normal distribution (with its variance equal to its standard deviation of one) would not be a great illustration of a key point. Instead, let's imagine we live in a town of 50,000 people and we know the height of everyone in this town. We will have 50,000 numbers that tell us everything about our population. We'll simulate these numbers now and put ourselves in one particular town, called 'town 47', where the population mean height is 172 cm and population standard deviation is 5 cm.

# In[28]:


seed(47)
pop_heights = norm.rvs(172, 5, size=50000)


# In[29]:


_ = plt.hist(pop_heights, bins=30)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in entire town population')
_ = plt.axvline(172, color='r')
_ = plt.axvline(172+5, color='r', linestyle='--')
_ = plt.axvline(172-5, color='r', linestyle='--')
_ = plt.axvline(172+10, color='r', linestyle='-.')
_ = plt.axvline(172-10, color='r', linestyle='-.')


# Now, 50,000 people is rather a lot to chase after with a tape measure. If all you want to know is the average height of the townsfolk, then can you just go out and measure a sample to get a pretty good estimate of the average height?

# In[30]:


def townsfolk_sampler(n):
    return np.random.choice(pop_heights, n)


# Let's say you go out one day and randomly sample 10 people to measure.

# In[31]:


seed(47)
daily_sample1 = townsfolk_sampler(10)


# In[32]:


_ = plt.hist(daily_sample1, bins=10)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in sample size 10')


# The sample distribution doesn't resemble what we take the population distribution to be. What do we get for the mean?

# In[33]:


np.mean(daily_sample1)


# And if we went out and repeated this experiment?

# In[34]:


daily_sample2 = townsfolk_sampler(10)


# In[35]:


np.mean(daily_sample2)


# __Q8:__ Simulate performing this random trial every day for a year, calculating the mean of each daily sample of 10, and plot the resultant sampling distribution of the mean.

# __A:__

# In[45]:


def trial_mean(days, ss):
    mean_list = []
    for i in range(days):
        sample_heights = np.random.choice(pop_heights, ss)
        sample_mean = np.mean(sample_heights)
        mean_list.append(sample_mean)
    return mean_list
    


# In[40]:


seed(47)
# take your samples here
mean_year = trial_mean(365, 10)
    


# In[41]:


_ = plt.hist(mean_year, bins=30)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of occurences')
_ = plt.title('Distribution of Sample Means After 365 Samples')


# The above is the distribution of the means of samples of size 10 taken from our population. The Central Limit Theorem tells us the expected mean of this distribution will be equal to the population mean, and standard deviation will be $\sigma / \sqrt n$, which, in this case, should be approximately 1.58.

# __Q9:__ Verify the above results from the CLT.

# __A:__ The mean of the repeated sampling distribution of the mean is converging on the true population mean of 172. The standard deviation of the sample distribution has the true population mean falling within it's bounds.

# In[43]:


np.mean(mean_year)


# In[44]:


np.std(mean_year, ddof=1)


# Remember, in this instance, we knew our population parameters, that the average height really is 172 cm and the standard deviation is 5 cm, and we see some of our daily estimates of the population mean were as low as around 168 and some as high as 176.

# __Q10:__ Repeat the above year's worth of samples but for a sample size of 50 (perhaps you had a bigger budget for conducting surveys that year)! Would you expect your distribution of sample means to be wider (more variable) or narrower (more consistent)? Compare your resultant summary statistics to those predicted by the CLT.

# __A:__

# In[13]:


seed(47)
# calculate daily means from the larger sample size here
larger_ss_means = trial_mean(365, 50)


# In[48]:


_ = plt.hist(larger_ss_means, bins=30)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of occurences')
_ = plt.title('Distribution of Sample Means After 365 Samples')


# What we've seen so far, then, is that we can estimate population parameters from a sample from the population, and that samples have their own distributions. Furthermore, the larger the sample size, the narrower are those sampling distributions.

# ### Normally testing time!

# All of the above is well and good. We've been sampling from a population we know is normally distributed, we've come to understand when to use $n$ and when to use $n-1$ in the denominator to calculate the spread of a distribution, and we've  seen the Central Limit Theorem in action for a sampling distribution. All seems very well behaved in Frequentist land. But, well, why should we really care?

# Remember, we rarely (if ever) actually know our population parameters but we still have to estimate them somehow. If we want to make inferences to conclusions like "this observation is unusual" or "my population mean has changed" then we need to have some idea of what the underlying distribution is so we can calculate relevant probabilities. In frequentist inference, we use the formulae above to deduce these population parameters. Take a moment in the next part of this assignment to refresh your understanding of how these probabilities work.

# Recall some basic properties of the standard normal distribution, such as that about 68% of observations are within plus or minus 1 standard deviation of the mean. Check out the precise definition of a normal distribution on p. 394 of *AoS*. 
# 
# __Q11:__ Using this fact, calculate the probability of observing the value 1 or less in a single observation from the standard normal distribution. Hint: you may find it helpful to sketch the standard normal distribution (the familiar bell shape) and mark the number of standard deviations from the mean on the x-axis and shade the regions of the curve that contain certain percentages of the population.

# __A:__ There is an 84.1% chance of observing a value <= 1

# Calculating this probability involved calculating the area under the curve from the value of 1 and below. To put it in mathematical terms, we need to *integrate* the probability density function. We could just add together the known areas of chunks (from -Inf to 0 and then 0 to $+\sigma$ in the example above). One way to do this is to look up tables (literally). Fortunately, scipy has this functionality built in with the cdf() function.

# __Q12:__ Use the cdf() function to answer the question above again and verify you get the same answer.

# __A:__

# In[50]:


norm.cdf(1)


# __Q13:__ Using our knowledge of the population parameters for our townsfolks' heights, what is the probability of selecting one person at random and their height being 177 cm or less? Calculate this using both of the approaches given above.

# __A:__ you have a 99.9% chance of selecting a person with a height of 177cm or less 

# In[58]:


norm.cdf((177-172)/1.58)


# __Q14:__ Turning this question around — suppose we randomly pick one person and measure their height and find they are 2.00 m tall. How surprised should we be at this result, given what we know about the population distribution? In other words, how likely would it be to obtain a value at least as extreme as this? Express this as a probability. 

# __A:__ It would be very suprising to select someone that is 200cm tall. This height value is over 5 standard deviations away from the mean and would be consided an extreme value with a probability of 0.000001

# In[59]:


norm.cdf((8/1.58))


# What we've just done is calculate the ***p-value*** of the observation of someone 2.00m tall (review *p*-values if you need to on p. 399 of *AoS*). We could calculate this probability by virtue of knowing the population parameters. We were then able to use the known properties of the relevant normal distribution to calculate the probability of observing a value at least as extreme as our test value.

# We're about to come to a pinch, though. We've said a couple of times that we rarely, if ever, know the true population parameters; we have to estimate them from our sample and we cannot even begin to estimate the standard deviation from a single observation. 
# 
# This is very true and usually we have sample sizes larger than one. This means we can calculate the mean of the sample as our best estimate of the population mean and the standard deviation as our best estimate of the population standard deviation. 
# 
# In other words, we are now coming to deal with the sampling distributions we mentioned above as we are generally concerned with the properties of the sample means we obtain. 
# 
# Above, we highlighted one result from the CLT, whereby the sampling distribution (of the mean) becomes narrower and narrower with the square root of the sample size. We remind ourselves that another result from the CLT is that _even if the underlying population distribution is not normal, the sampling distribution will tend to become normal with sufficiently large sample size_. (**Check out p. 199 of AoS if you need to revise this**). This is the key driver for us 'requiring' a certain sample size, for example you may frequently see a minimum sample size of 30 stated in many places. In reality this is simply a rule of thumb; if the underlying distribution is approximately normal then your sampling distribution will already be pretty normal, but if the underlying distribution is heavily skewed then you'd want to increase your sample size.

# __Q15:__ Let's now start from the position of knowing nothing about the heights of people in our town.
# * Use the random seed of 47, to randomly sample the heights of 50 townsfolk
# * Estimate the population mean using np.mean
# * Estimate the population standard deviation using np.std (remember which denominator to use!)
# * Calculate the (95%) [margin of error](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/margin-of-error/#WhatMofE) (use the exact critial z value to 2 decimal places - [look this up](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/find-critical-values/) or use norm.ppf()) Recall that the ***margin of error*** is mentioned on p. 189 of the *AoS* and discussed in depth in that chapter). 
# * Calculate the 95% Confidence Interval of the mean (***confidence intervals*** are defined on p. 385 of *AoS*) 
# * Does this interval include the true population mean?

# __A:__

# In[4]:


seed(47)
# take your sample now


# In[7]:


sample_50 = norm.rvs(size=50)


# In[8]:


np.mean(sample_50)


# In[9]:


np.std(sample_50, ddof=1)


# In[11]:


1.96 * np.std(sample_50, ddof=1)


# 95 confidence interval = -0.18 + or - 0.97

# The true population mean falls into this range

# __Q16:__ Above, we calculated the confidence interval using the critical z value. What is the problem with this? What requirement, or requirements, are we (strictly) failing?

# __A:__ If we dont have a sufficient sample size or know the population standard deviation we should use a t-scpre

# __Q17:__ Calculate the 95% confidence interval for the mean using the _t_ distribution. Is this wider or narrower than that based on the normal distribution above? If you're unsure, you may find this [resource](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/confidence-interval/) useful. For calculating the critical value, remember how you could calculate this for the normal distribution using norm.ppf().

# __A:__ using the t-score provides a wider margin of error 

# In[12]:


2.010 * np.std(sample_50, ddof=1)


# In[17]:


norm.ppf(0.95, loc=-.181, scale=1.91)


# In[ ]:





# This is slightly wider than the previous confidence interval. This reflects the greater uncertainty given that we are estimating population parameters from a sample.

# ## 4. Learning outcomes

# Having completed this project notebook, you now have hands-on experience:
# * sampling and calculating probabilities from a normal distribution
# * identifying the correct way to estimate the standard deviation of a population (the population parameter) from a sample
# * with sampling distribution and now know how the Central Limit Theorem applies
# * with how to calculate critical values and confidence intervals

# In[ ]:




