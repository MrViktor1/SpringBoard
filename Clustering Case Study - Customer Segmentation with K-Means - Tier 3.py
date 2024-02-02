#!/usr/bin/env python
# coding: utf-8

# # Clustering Case Study: Customer Segmentation with K-Means -  Tier 3
# ***
# This case study is based on [this blog post](http://blog.yhat.com/posts/customer-segmentation-using-python.html) by the `yhat` blog. Please feel free to refer to the post for additional information, and solutions.
# 
# Structure of the mini-project:
# 
# 1. **Sourcing and loading**
#     * Load the data
#     * Explore the data
# 
# 
# 2. **Cleaning, transforming and visualizing**
#     * Data Wrangling: Exercise Set 1
#         - Creating a matrix with a binary indicator for whether they responded to a given offer
#         - Ensure that in doing so, NAN values are dealt with appropriately
#     
# 
# 3. **Modelling** 
#     * K-Means clustering: Exercise Sets 2 and 3
#         - Choosing K: The Elbow method
#         - Choosing K: The Silhouette method
#         - Choosing K: The Gap statistic method
#     
#     * Visualizing clusters with PCA: Exercise Sets 4 and 5
# 
# 
# 4. **Conclusions and next steps**
#     * Conclusions
#     * Other clustering algorithms (Exercise Set 6)

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Seaborn
sns.set_style("whitegrid")
sns.set_context("poster")


# ## 1. Sourcing and loading
# ### 1a. Load the data
# The dataset contains information on marketing newsletters/e-mail campaigns (e-mail offers sent to customers) and transaction level data from customers. The transactional data shows which offer customers responded to, and what the customer ended up buying. The data is presented as an Excel workbook containing two worksheets. Each worksheet contains a different dataset.

# In[2]:


df_offers = pd.read_excel("./WineKMC.xlsx", sheet_name=0)


# ### 1b. Explore the data

# In[3]:


df_offers.columns = ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]
df_offers.head()


# We see that the first dataset contains information about each offer such as the month it is in effect and several attributes about the wine that the offer refers to: the variety, minimum quantity, discount, country of origin and whether or not it is past peak. The second dataset in the second worksheet contains transactional data -- which offer each customer responded to.

# In[4]:


df_transactions = pd.read_excel("./WineKMC.xlsx", sheet_name=1)
df_transactions.columns = ["customer_name", "offer_id"]
df_transactions['n'] = 1
df_transactions.head()


# ## 2. Cleaning, transforming and visualizing
# ### 2a. Data Wrangling

# We're trying to learn more about how our customers behave, so we can use their behavior (whether or not they purchased something based on an offer) as a way to group similar minded customers together. We can then study those groups to look for patterns and trends which can help us formulate future offers.
# 
# The first thing we need is a way to compare customers. To do this, we're going to create a matrix that contains each customer and a 0/1 indicator for whether or not they responded to a given offer. 

# # <div class="span5 alert alert-info">
# <h3>Checkup Exercise Set I</h3>
# 
# <p><b>Exercise:</b> Create a data frame where each row has the following columns (Use the pandas [`merge`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html) and [`pivot_table`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html) functions for this purpose):
# <ul>
# <li> customer_names
# <li> One column for each offer, with a 1 if the customer responded to the offer
# </ul>
# <p>Make sure you also deal with any weird values such as `NaN`. Read the documentation to develop your solution.</p>
# </div>

# In[5]:


#your turn
#merge offer df and tansaction df on offer_id
df_response = pd.merge(df_transactions, df_offers)


# In[6]:


# turn the values of offer response into a pivot table
pivot_response = df_response.pivot_table(index='customer_name', columns='offer_id', values='n', fill_value=0)

pivot_response


# ## 3. Modelling 
# ### 3a. K-Means Clustering
# 
# Recall that in K-Means Clustering we want to *maximize* the distance between centroids and *minimize* the distance between data points and the respective centroid for the cluster they are in. True evaluation for unsupervised learning would require labeled data; however, we can use a variety of intuitive metrics to try to pick the number of clusters K. We will introduce two methods: the Elbow method, the Silhouette method and the gap statistic.

# #### 3ai. Choosing K: The Elbow Sum-of-Squares Method
# 
# The first method looks at the sum-of-squares error in each cluster against $K$. We compute the distance from each data point to the center of the cluster (centroid) to which the data point was assigned. 
# 
# $$SS = \sum_k \sum_{x_i \in C_k} \sum_{x_j \in C_k} \left( x_i - x_j \right)^2 = \sum_k \sum_{x_i \in C_k} \left( x_i - \mu_k \right)^2$$
# 
# where $x_i$ is a point, $C_k$ represents cluster $k$ and $\mu_k$ is the centroid for cluster $k$. We can plot SS vs. $K$ and choose the *elbow point* in the plot as the best value for $K$. The elbow point is the point at which the plot starts descending much more slowly. 
# 
# **Hint:** the Elbow Method is discussed in part 2 of the Harvard Clustering lecture. 

# <div class="span5 alert alert-info">
# <h3>Checkup Exercise Set II</h3>
# 
# <p><b>Exercise:</b></p> 
# <ul>
# <li> What values of $SS$ do you believe represent better clusterings? Why?
# <li> Create a numpy matrix `x_cols` with only the columns representing the offers (i.e. the 0/1 colums) 
# <li> Write code that applies the [`KMeans`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) clustering method from scikit-learn to this matrix. 
# <li> Construct a plot showing $SS$ for each $K$ and pick $K$ using this plot. For simplicity, test $2 \le K \le 10$.
# <li> Make a bar chart showing the number of points in each cluster for k-means under the best $K$.
# <li> What challenges did you experience using the Elbow method to pick $K$?
# </ul>
# </div>

# In[7]:


# your turn
# my intuiton is k=2 is the best as there are 2 main types of wine drinkers in my mind, casual and fine.
# small numbers for ss will indicate better fit clusterings as the distance between the centroids and the data is smallest.
import sklearn.cluster
import numpy as np

#make a empyt list for sum of squares
ss = []

#make empty dict
assignments = {}

#cast table 
x = pivot_response.to_numpy()

#make a list of k values
k_val = list(range(2,11))


#loop through kmeans fit and model with k values and put outputs in the list 
for i in k_val:
    model = sklearn.cluster.KMeans(n_clusters= i)
    assigned_cluster = model.fit_predict(x)
    centers = model.cluster_centers_
    ss.append(np.sum((x - centers[assigned_cluster]) ** 2))
    assignments[str(i)] = assigned_cluster
    
#plot different values of k
plt.plot(k_val, ss)


# In[8]:


# What is the best K? Fill in the assignment below appropriately
best_K = 4
assignments_best_K = assignments[str(best_K)]
counts = np.bincount(assignments_best_K)
print(len(counts))

# Call bar() on plt, with parameters range(best_K), counts, and align = 'center'
plt.bar(range(best_K), counts, align = 'center')

# Label the axes 
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.xticks(range(best_K))


# #### 3aii. Choosing K: The Silhouette Method
# 
# There exists another method that measures how well each datapoint $x_i$ "fits" its assigned cluster *and also* how poorly it fits into other clusters. This is a different way of looking at the same objective. Denote $a_{x_i}$ as the *average* distance from $x_i$ to all other points within its own cluster $k$. The lower the value, the better. On the other hand $b_{x_i}$ is the minimum average distance from $x_i$ to points in a different cluster, minimized over clusters. That is, compute separately for each cluster the average distance from $x_i$ to the points within that cluster, and then take the minimum. The silhouette $s(x_i)$ is defined as
# 
# $$s(x_i) = \frac{b_{x_i} - a_{x_i}}{\max{\left( a_{x_i}, b_{x_i}\right)}}$$
# 
# The silhouette score is computed on *every datapoint in every cluster*. The silhouette score ranges from -1 (a poor clustering) to +1 (a very dense clustering) with 0 denoting the situation where clusters overlap. Some criteria for the silhouette coefficient is provided in the table below.

# <pre>
# 
# | Range       | Interpretation                                |
# |-------------|-----------------------------------------------|
# | 0.71 - 1.0  | A strong structure has been found.            |
# | 0.51 - 0.7  | A reasonable structure has been found.        |
# | 0.26 - 0.5  | The structure is weak and could be artificial.|
# | < 0.25      | No substantial structure has been found.      |
# 
# </pre>
# Source: http://www.stat.berkeley.edu/~spector/s133/Clus.html

# **Hint**: Scikit-learn provides a function to compute this for us (phew!) called [`sklearn.metrics.silhouette_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html). Take a look at [this article](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) on picking $K$ in scikit-learn, as it will help you in the next exercise set.

# <div class="span5 alert alert-info">
# <h3>Checkup Exercise Set III</h3>
# 
# <p><b>Exercise:</b> Using the documentation for the `silhouette_score` function above, construct a series of silhouette plots like the ones in the article linked above.</p>
# 
# <p><b>Exercise:</b> Compute the average silhouette score for each $K$ and plot it. What $K$ does the plot suggest we should choose? Does it differ from what we found using the Elbow method?</p>
# </div>

# In[9]:


import sklearn.metrics
import matplotlib.cm as cm

# Make an empty list
avg_silhouette_scores = []

# Iterate through Krange 
for K in k_val:
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 5)

    # Call set_xlim on ax1 
    ax1.set_xlim([-0.25, 1])
    # plots of individual clusters
    ax1.set_ylim([0, len(x) + (K + 1) * 10])

    # Initialize the clusterer
    clusterer = sklearn.cluster.KMeans(n_clusters=K, random_state=10)
    cluster_labels = clusterer.fit_predict(x)

    silhouette_avg = sklearn.metrics.silhouette_score(x, cluster_labels)
    avg_silhouette_scores.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = sklearn.metrics.silhouette_samples(x, cluster_labels)

    y_lower = 10
    for i in range(K):
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        # Call sort() on this variable
        ith_cluster_silhouette_values.sort()
    
        # Call shape[0] on ith_cluster_silhouette_values 
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / K)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower
    y_lower = y_upper + 10  

    # Setting title, xlabel and ylabel 
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % K),
                 fontsize=14, fontweight='bold')


# In[10]:


plt.plot(k_val, avg_silhouette_scores)
plt.xlabel("$K$")
plt.ylabel("Average Silhouette Score")


# This graph suggest a k of 5 has the highest average silhouette score. This differs from the elbow method where 4 was the selection for best value of k.

# #### 3aiii.  Choosing $K$: The Gap Statistic
# 
# There is one last method worth covering for picking $K$, the so-called Gap statistic. The computation for the gap statistic builds on the sum-of-squares established in the Elbow method discussion, and compares it to the sum-of-squares of a "null distribution," that is, a random set of points with no clustering. The estimate for the optimal number of clusters $K$ is the value for which $\log{SS}$ falls the farthest below that of the reference distribution:
# 
# $$G_k = E_n^*\{\log SS_k\} - \log SS_k$$
# 
# In other words a good clustering yields a much larger difference between the reference distribution and the clustered data. The reference distribution is a Monte Carlo (randomization) procedure that constructs $B$ random distributions of points within the bounding box (limits) of the original data and then applies K-means to this synthetic distribution of data points.. $E_n^*\{\log SS_k\}$ is just the average $SS_k$ over all $B$ replicates. We then compute the standard deviation $\sigma_{SS}$ of the values of $SS_k$ computed from the $B$ replicates of the reference distribution and compute
# 
# $$s_k = \sqrt{1+1/B}\sigma_{SS}$$
# 
# Finally, we choose $K=k$ such that $G_k \geq G_{k+1} - s_{k+1}$.

# #### Aside: Choosing $K$ when we Have Labels
# 
# Unsupervised learning expects that we do not have the labels. In some situations, we may wish to cluster data that is labeled. Computing the optimal number of clusters is much easier if we have access to labels. There are several methods available. We will not go into the math or details since it is rare to have access to the labels, but we provide the names and references of these measures.
# 
# * Adjusted Rand Index
# * Mutual Information
# * V-Measure
# * Fowlkes–Mallows index
# 
# **Hint:** See [this article](http://scikit-learn.org/stable/modules/clustering.html) for more information about these metrics.

# ### 3b. Visualizing Clusters using PCA
# 
# How do we visualize clusters? If we only had two features, we could likely plot the data as is. But we have 100 data points each containing 32 features (dimensions). Principal Component Analysis (PCA) will help us reduce the dimensionality of our data from 32 to something lower. For a visualization on the coordinate plane, we will use 2 dimensions. In this exercise, we're going to use it to transform our multi-dimensional dataset into a 2 dimensional dataset.
# 
# This is only one use of PCA for dimension reduction. We can also use PCA when we want to perform regression but we have a set of highly correlated variables. PCA untangles these correlations into a smaller number of features/predictors all of which are orthogonal (not correlated). PCA is also used to reduce a large set of variables into a much smaller one.
# 
# **Hint:** PCA was discussed in the previous subunit. If you need help with it, consult [this useful article](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c) and [this visual explanation](http://setosa.io/ev/principal-component-analysis/). 

# #<div class="span5 alert alert-info">
# <h3>Checkup Exercise Set IV</h3>
# 
# <p><b>Exercise:</b> Use PCA to plot your clusters:</p>
# 
# <ul>
# <li> Use scikit-learn's [`PCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) function to reduce the dimensionality of your clustering data to 2 components
# <li> Create a data frame with the following fields:
#   <ul>
#   <li> customer name
#   <li> cluster id the customer belongs to
#   <li> the two PCA components (label them `x` and `y`)
#   </ul>
# <li> Plot a scatterplot of the `x` vs `y` columns
# <li> Color-code points differently based on cluster ID
# <li> How do the clusters look? 
# <li> Based on what you see, what seems to be the best value for $K$? Moreover, which method of choosing $K$ seems to have produced the optimal result visually?
# </ul>
# 
# <p><b>Exercise:</b> Now look at both the original raw data about the offers and transactions and look at the fitted clusters. Tell a story about the clusters in context of the original data. For example, do the clusters correspond to wine variants or something else interesting?</p>
# </div>

# In[11]:


#your turn
import sklearn.decomposition
import matplotlib.colors

model = sklearn.cluster.KMeans(n_clusters=3)

cluster_assignments = model.fit_predict(x)

cmap = matplotlib.colors.ListedColormap(['red', 'green', 'blue'])

pca = sklearn.decomposition.PCA(n_components=2)
pc1, pc2 = zip(*pca.fit_transform(x))
plt.scatter(pc1, pc2, c=cluster_assignments.tolist(), cmap=cmap)


# In[12]:


model = sklearn.cluster.KMeans(n_clusters=3)
cluster_assignments = model.fit_predict(x)

colors = ['red', 'green', 'blue']
offer_proportions = pivot_response.sum(axis=0) / 100  # There are 100 customers
for i in range(3):
    plt.figure(i)
    cluster = pivot_response[cluster_assignments == i]
    offer_proportions_cluster = cluster.sum(axis=0) / cluster.shape[0]  # Number of customers in cluster
    lift = offer_proportions_cluster - offer_proportions
    plt.bar(range(1, 33), lift, color=colors[i])


# Based on the pca analysis the best choice for k is 3. Based off all the methods of selecting k that were used I beieve k = 3 is the best choice for this data set. The silhouette and pca analysis showed this the clearest. The grouping of 3 may be fitting for the major red, white, and champagne varieties that wine is offered in.

# What we've done is we've taken those columns of 0/1 indicator variables, and we've transformed them into a 2-D dataset. We took one column and arbitrarily called it `x` and then called the other `y`. Now we can throw each point into a scatterplot. We color coded each point based on it's cluster so it's easier to see them.

# <div class="span5 alert alert-info">
# <h3>Exercise Set V</h3>
# 
# <p>As we saw earlier, PCA has a lot of other uses. Since we wanted to visualize our data in 2 dimensions, restricted the number of dimensions to 2 in PCA. But what is the true optimal number of dimensions?</p>
# 
# <p><b>Exercise:</b> Using a new PCA object shown in the next cell, plot the `explained_variance_` field and look for the elbow point, the point where the curve's rate of descent seems to slow sharply. This value is one possible value for the optimal number of dimensions. What is it?</p>
# </div>

# In[13]:


#your turn
# Initialize a new PCA model with a default number of components.
import sklearn.decomposition
pca = sklearn.decomposition.PCA()
pca.fit(x)

# Do the rest on your own :)
variance = pca.explained_variance_ratio_

plt.plot(range(len(variance)), variance)

plt.xlabel("number of components")
plt.ylabel("prportion of variance explained")


# ## 4. Conclusions and next steps
# ### 4a. Conclusions
# What can you conclude from your investigations? Make a note, formulate it as clearly as possible, and be prepared to discuss it with your mentor in your next call. 

# the elbow in this graph appears to be 3 or 4. 
# The conclusion I can make from this dimensional analysis is that more product knowledge would be helpfull in determining a proper k value. The k value seems to be best when limited between 3 and 5 and could  be explained by wine variations or different seasons in which the sales took place.

# ### 4b. Other clustering algorithms
# 
# k-means is only one of a ton of clustering algorithms. Below is a brief description of several clustering algorithms, and the table provides references to the other clustering algorithms in scikit-learn. 
# 
# * **Affinity Propagation** does not require the number of clusters $K$ to be known in advance! AP uses a "message passing" paradigm to cluster points based on their similarity. 
# 
# * **Spectral Clustering** uses the eigenvalues of a similarity matrix to reduce the dimensionality of the data before clustering in a lower dimensional space. This is tangentially similar to what we did to visualize k-means clusters using PCA. The number of clusters must be known a priori.
# 
# * **Ward's Method** applies to hierarchical clustering. Hierarchical clustering algorithms take a set of data and successively divide the observations into more and more clusters at each layer of the hierarchy. Ward's method is used to determine when two clusters in the hierarchy should be combined into one. It is basically an extension of hierarchical clustering. Hierarchical clustering is *divisive*, that is, all observations are part of the same cluster at first, and at each successive iteration, the clusters are made smaller and smaller. With hierarchical clustering, a hierarchy is constructed, and there is not really the concept of "number of clusters." The number of clusters simply determines how low or how high in the hierarchy we reference and can be determined empirically or by looking at the [dendogram](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.cluster.hierarchy.dendrogram.html).
# 
# * **Agglomerative Clustering** is similar to hierarchical clustering but but is not divisive, it is *agglomerative*. That is, every observation is placed into its own cluster and at each iteration or level or the hierarchy, observations are merged into fewer and fewer clusters until convergence. Similar to hierarchical clustering, the constructed hierarchy contains all possible numbers of clusters and it is up to the analyst to pick the number by reviewing statistics or the dendogram.
# 
# * **DBSCAN** is based on point density rather than distance. It groups together points with many nearby neighbors. DBSCAN is one of the most cited algorithms in the literature. It does not require knowing the number of clusters a priori, but does require specifying the neighborhood size.

# ### Clustering Algorithms in Scikit-learn
# <table border="1">
# <colgroup>
# <col width="15%" />
# <col width="16%" />
# <col width="20%" />
# <col width="27%" />
# <col width="22%" />
# </colgroup>
# <thead valign="bottom">
# <tr><th>Method name</th>
# <th>Parameters</th>
# <th>Scalability</th>
# <th>Use Case</th>
# <th>Geometry (metric used)</th>
# </tr>
# </thead>
# <tbody valign="top">
# <tr><td>K-Means</span></a></td>
# <td>number of clusters</td>
# <td>Very large<span class="pre">n_samples</span>, medium <span class="pre">n_clusters</span> with
# MiniBatch code</td>
# <td>General-purpose, even cluster size, flat geometry, not too many clusters</td>
# <td>Distances between points</td>
# </tr>
# <tr><td>Affinity propagation</td>
# <td>damping, sample preference</td>
# <td>Not scalable with n_samples</td>
# <td>Many clusters, uneven cluster size, non-flat geometry</td>
# <td>Graph distance (e.g. nearest-neighbor graph)</td>
# </tr>
# <tr><td>Mean-shift</td>
# <td>bandwidth</td>
# <td>Not scalable with <span class="pre">n_samples</span></td>
# <td>Many clusters, uneven cluster size, non-flat geometry</td>
# <td>Distances between points</td>
# </tr>
# <tr><td>Spectral clustering</td>
# <td>number of clusters</td>
# <td>Medium <span class="pre">n_samples</span>, small <span class="pre">n_clusters</span></td>
# <td>Few clusters, even cluster size, non-flat geometry</td>
# <td>Graph distance (e.g. nearest-neighbor graph)</td>
# </tr>
# <tr><td>Ward hierarchical clustering</td>
# <td>number of clusters</td>
# <td>Large <span class="pre">n_samples</span> and <span class="pre">n_clusters</span></td>
# <td>Many clusters, possibly connectivity constraints</td>
# <td>Distances between points</td>
# </tr>
# <tr><td>Agglomerative clustering</td>
# <td>number of clusters, linkage type, distance</td>
# <td>Large <span class="pre">n_samples</span> and <span class="pre">n_clusters</span></td>
# <td>Many clusters, possibly connectivity constraints, non Euclidean
# distances</td>
# <td>Any pairwise distance</td>
# </tr>
# <tr><td>DBSCAN</td>
# <td>neighborhood size</td>
# <td>Very large <span class="pre">n_samples</span>, medium <span class="pre">n_clusters</span></td>
# <td>Non-flat geometry, uneven cluster sizes</td>
# <td>Distances between nearest points</td>
# </tr>
# <tr><td>Gaussian mixtures</td>
# <td>many</td>
# <td>Not scalable</td>
# <td>Flat geometry, good for density estimation</td>
# <td>Mahalanobis distances to  centers</td>
# </tr>
# <tr><td>Birch</td>
# <td>branching factor, threshold, optional global clusterer.</td>
# <td>Large <span class="pre">n_clusters</span> and <span class="pre">n_samples</span></td>
# <td>Large dataset, outlier removal, data reduction.</td>
# <td>Euclidean distance between points</td>
# </tr>
# </tbody>
# </table>
# Source: http://scikit-learn.org/stable/modules/clustering.html

# <div class="span5 alert alert-info">
# <h3>Exercise Set VI</h3>
# 
# <p><b>Exercise:</b> Try clustering using the following algorithms. </p>
# <ol>
# <li>Affinity propagation
# <li>Spectral clustering
# <li>Agglomerative clustering
# <li>DBSCAN
# </ol>
# <p>How do their results compare? Which performs the best? Tell a story why you think it performs the best.</p>
# </div>
# 

# Because I don't have a good estimate for the value of K I think Affinity propagation or DBSCAN would work best in this case.
# Affinity propigation and DBSCAN have the benifit of working with uneven sized clusters which may be relivent for this dataset. DBSCAN works best with datasets with large numbers of samples which is not ideal for the current dataset.

# In[23]:


from sklearn.cluster import AffinityPropagation
from sklearn import metrics

clustering = AffinityPropagation(random_state=10).fit(x)
cluster_centers_indices = clustering.cluster_centers_indices_
labels = clustering.labels_

n_clusters_ = len(cluster_centers_indices)

print("Estimated number of clusters: %d" % n_clusters_)
print(
    "Silhouette Coefficient: %0.3f"
    % metrics.silhouette_score(x, labels, metric="sqeuclidean")
)


# In[38]:


from sklearn.cluster import DBSCAN

db = DBSCAN(eps=.5, min_samples=3).fit(x)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


# In[39]:


unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = x[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = x[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()

