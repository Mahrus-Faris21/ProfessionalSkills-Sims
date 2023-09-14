#!/usr/bin/env python
# coding: utf-8

# # Background Information About the Task

# # Case 4: Display Information

# Now that you are familiar with the portfolio and personal loans and risk are using your model as a guide to loss provisions for the upcoming year, the team now asks you to look at their mortgage book. They suspect that FICO scores will provide a good indication of how likely a customer is to default on their mortgage. Charlie wants to build a machine learning model that will predict the probability of default, but while you are discussing the methodology, she mentions that the architecture she is using requires categorical data. As FICO ratings can take integer values in a large range, they will need to be mapped into buckets. She asks if you can find the best way of doing this to allow her to analyze the data.
# 
# A FICO score is a standardized credit score created by the Fair Isaac Corporation (FICO) that quantifies the creditworthiness of a borrower to a value between 300 to 850, based on various factors. FICO scores are used in 90% of mortgage application decisions in the United States. The risk manager provides you with FICO scores for the borrowers in the bank’s portfolio and wants you to construct a technique for predicting the PD (probability of default) for the borrowers using these scores. 

# # Case 4: Task Brief

# Charlie wants to make her model work for future data sets, so she needs a general approach to generating the buckets. Given a set number of buckets corresponding to the number of input labels for the model, she would like to find out the boundaries that best summarize the data. You need to create a rating map that maps the FICO score of the borrowers to a rating where a lower rating signifies a better credit score.
# 
# The process of doing this is known as quantization. You could consider many ways of solving the problem by optimizing different properties of the resulting buckets, such as the mean squared error or log-likelihood (see below for definitions). For background on quantization, see here.
# 
# Mean squared error:
# 
# You can view this question as an approximation problem and try to map all the entries in a bucket to one value, minimizing the associated squared error. We are now looking to minimize the following:
# $$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2$$
# Log-likelihood:
# 
# A more sophisticated possibility is to maximize the following log-likelihood function.
# $$\log L(\theta|y) = \sum_{i=1}^{n}\log f(y_i|\theta)$$
# Where $b_i$ is the bucket boundaries, ni is the number of records in each bucket, $k_i$ is the number of defaults in each bucket, and $p_i = k_i / n_i$ is the probability of default in the bucket. This function considers how rough the discretization is and the density of defaults in each bucket. This problem could be addressed by splitting it into subproblems, which can be solved incrementally (i.e., through a dynamic programming approach). For example, you can break the problem into two subproblems, creating five buckets for FICO scores ranging from 0 to 600 and five buckets for FICO scores ranging from 600 to 850. Refer to this page for more context behind a likelihood function. This page may also be helpful for background on dynamic programming. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


csv_path = r'C:\Users\Sweet\Downloads\Portofolio\Task 3 and 4_Loan_Data.csv'
data = pd.read_csv(csv_path)
data.sort_index(inplace=True)
# Define the boundaries for the FICO score intervals
fico_boundaries = [300, 500, 600, 700, 800, 850]

# Define the rating categories
rating_categories = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

# Create a new column for the rating
data['FICO Rating'] = pd.cut(data['fico_score'], bins=fico_boundaries, labels=rating_categories)

# Print the first few rows of the data with the rating column
print(data.head())


# # Determine The FICO Ratings

# In[ ]:


data.sort_index(inplace=True)

# Define the number of intervals
num_intervals = 32

# Define the range of FICO scores
fico_min = data['fico_score'].min()
fico_max = data['fico_score'].max()

# Initialize the cost matrix
cost_matrix = np.zeros((num_intervals, len(data)))

# Initialize the boundary matrix
boundary_matrix = np.zeros((num_intervals-1, len(data)))

# Initialize the first row of the cost matrix
for i in range(len(data)):
    cost_matrix[0, i] = np.var(data['fico_score'][:i+1])

# Fill in the rest of the cost matrix and boundary matrix
for i in range(1, num_intervals):
    for j in range(i, len(data)):
        min_cost = np.inf
        for k in range(i-1, j):
            cost = cost_matrix[i-2, k] + np.var(data['fico_score'][k+1:j+1])
            if cost < min_cost:
                min_cost = cost
                boundary_matrix[i-1, j-1] = k
        cost_matrix[i-1, j] = min_cost

# Find the optimal boundaries
boundaries = [fico_min]
for i in range(num_intervals-1):
    boundaries.append(data['fico_score'][int(boundary_matrix[i, len(data)-1])])
boundaries.append(fico_max)

# Define the rating categories
rating_categories = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

# Create a new column for the rating
data['FICO Rating'] = pd.cut(data['fico_score'], bins=boundaries, labels=rating_categories)

# Print the first few rows of the data with the rating column
print(data.head())


# # Determine of Log-likelihood and Quantization Signal Processing

# In[6]:


def log_likelihood(x, mu, sigma):
    return np.sum(np.log(np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)))

def quantize_signal(x, k):
    # Estimate the parameters of the PDF
    mu = np.mean(x)
    sigma = np.std(x)
    
    # Divide the range of the data into k intervals
    boundaries = np.linspace(np.min(x), np.max(x), k+1)
    
    # Calculate the log-likelihood of the data for each interval
    ll = np.zeros(k)
    for i in range(k):
        ll[i] = log_likelihood(x[(x >= boundaries[i]) & (x < boundaries[i+1])], mu, sigma)
    
    # Assign a quantization index to each interval based on the log-likelihood
    indices = np.argsort(ll)[::-1]
    quantized = np.zeros(len(x))
    for i in range(k):
        quantized[(x >= boundaries[indices[i]]) & (x < boundaries[indices[i]+1])] = i
    
    # Use dynamic programming to optimize the boundaries of the intervals
    cost = np.zeros((len(x), k))
    for j in range(k):
        for i in range(j, len(x)):
            if j == 0:
                cost[i,j] = 0
            else:
                cost[i,j] = np.inf
                for k in range(j-1, i):
                    c = cost[k,j-1] + log_likelihood(x[k+1:i+1], np.mean(x[k+1:i+1]), np.std(x[k+1:i+1]))
                    if c < cost[i,j]:
                        cost[i,j] = c
    
    # Find the optimal boundaries
    optimal_boundaries = np.zeros(k-1)
    j = k-1
    i = len(x)-1
    while j > 0:
        for k in range(j-1, i):
            if cost[i,j] == cost[k,j-1] + log_likelihood(x[k+1:i+1], np.mean(x[k+1:i+1]), np.std(x[k+1:i+1])):
                optimal_boundaries[j-1] = boundaries[k+1]
                i = k
                j -= 1
                break
    
    return quantized, optimal_boundaries


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

# Define the features and target variable
X = data[['fico_score']]
y = data['default']

# Fit a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict the probability of default for each borrower
y_pred = model.predict_proba(X)[:, 1]

# Calculate the MSE
mse = mean_squared_error(y, y_pred)

# Calculate the log-likelihood
log_likelihood = np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

# Print the MSE and log-likelihood
print('MSE:', mse)
print('Log-likelihood:', log_likelihood)


# To prepare the FICO scores for use in Charlie's machine learning model, we employ a technique known as quantization, which involves mapping the FICO scores into categorical buckets. The objective is to transform the continuous FICO scores into discrete categories to enable the model's analysis.
# 
# The optimization criterion for determining the bucket boundaries typically involves minimizing Mean Squared Error (MSE) or maximizing Log-likelihood. These criteria help identify the boundaries that best summarize the data while maintaining the model's predictive accuracy.
# 
# For our specific dataset, the quantization process resulted in the following metrics:
# 
# Mean Squared Error (MSE): 0.1328
# Log-likelihood: -4239.15
# 
# These metrics provide insights into the quality of the bucket boundaries chosen for mapping FICO scores. A lower MSE and a higher log-likelihood indicate a better fit of the bucket boundaries to the data.
# 
# By implementing this quantization technique, we can effectively prepare the FICO scores as categorical data for Charlie's machine learning model, allowing her to analyze and predict the probability of default for mortgage borrowers.
# 

# In[4]:


fico_data = data['fico_score'].values

mu = np.mean(fico_data)
sigma = np.std(fico_data)
p_range = np.linspace(mu-3*sigma, mu+3*sigma, 100)

# Calculate the log-likelihood for each parameter value
ll = np.zeros(len(p_range))
for i in range(len(p_range)):
    ll[i] = np.sum(np.log(np.exp(-(fico_data-p_range[i])**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)))
    
plt.plot(p_range, ll)
plt.xlabel('Parameter')
plt.ylabel('Log-likelihood')
plt.title('Log-likelihood function for FICO scores')
plt.show()

