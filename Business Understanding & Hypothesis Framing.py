#!/usr/bin/env python
# coding: utf-8

# # Here is the background information on your task
# PowerCo is a major gas and electricity utility that supplies to corporate, SME (Small & Medium Enterprise), and residential customers. The power-liberalization of the energy market in Europe has led to significant customer churn, especially in the SME segment. They have partnered with BCG to help diagnose the source of churning SME customers.
# A fair hypothesis is that price changes affect customer churn. Therefore, it is helpful to know which customers are more (or less) likely to churn at their current price, for which a good predictive model could be useful. 
# Moreover, for those customers that are at risk of churning, a discount might incentivize them to stay with our client. The head of the SME division is considering a 20% discount that is considered large enough to dissuade almost anyone from churning (especially those for whom price is the primary concern).
# The Associate Director (AD) held an initial team meeting to discuss various hypotheses, including churn due to price sensitivity. After discussion with your team, you have been asked to go deeper on the hypothesis that the churn is driven by the customers’ price sensitivities.  
# Your AD wants an email with your thoughts on how the team should go about testing this hypothesis.
# The client plans to use the predictive model on the 1st working day of every month to indicate to which customers the 20% discount should be offered.

# # Here is your task
# Your first task today is to understand what is going on with the client and to think about how you would approach this problem and test the specific hypothesis.
# You must formulate the hypothesis as a data science problem and lay out the major steps needed to test this hypothesis. Communicate your thoughts and findings in an email to your AD, focusing on the data that you would need from the client and the analytical models you would use to test such a hypothesis.
# 
# We would suggest spending no more than one hour on this task.
# Please note, there are multiple ways to approach the task and that the model answer is just one way to do it.
# 
# **If you are stuck:**
# - Remember what the key factors are for a customer deciding to stay with or switch providers
# - Think of data sources and fields that could be used to explore the contribution of various factors to a customer’s possible action 
# - Ideally, what would a data frame of your choice look like – what should each column and row represent? 
# - What kind of exploratory analyses on the relevant fields can give more insights about the customer's churn behavior? 
# 
# **Estimated time for task completion: 1 hour depending on your learning style.**
# # The Answers

# To proem the answers that require from above tasks, I have recently to set up the operating libraries that would to support the variety questions that involved in mentioned from my client. As the closer I presented the steps into from following this section:
# **Step 1: Defining the Problem**
# I will define the problem as a task of classifying whether a customer is likely to leave or stay based on the provided data. In this case, the 'churn' variable is what I'm trying to predict, and the other columns in the dataset are the factors I'll use for making this prediction. The hypothesis I'm testing is whether customer sensitivity to prices, represented by factors like 'cons_12m,' 'imp_cons,' and others, affects the likelihood of them leaving.
# 
# **Step 2: Gathering Data**
# I will get in touch with the client to collect the necessary data. You mentioned that there's already a dataset with columns like 'cons_12m,' 'imp_cons,' and others, which seems like a good starting point.
# 
# **Step 3: Data Preparation**
# I will clean and prepare the data. This might involve handling missing information, converting date columns to the right date-time format, encoding categorical variables like 'channel_sales' and 'origin_up,' and scaling or normalizing numerical characteristics if necessary.
# 
# **Step 4: Exploring the Data**
# I will conduct an initial analysis of the data to get a better understanding. Some things I can do during this phase include:
# - Describing the data using statistics (average, median, standard deviation, etc.) for relevant columns.
# - Creating visual representations like histograms and box plots to see how the data is distributed.
# - Examining correlations between the variables and the 'churn' target variable.
# 
# **Step 5: Creating New Features**
# I will generate meaningful features that could assist in predicting churn. This might involve developing entirely new features or transforming existing ones. For instance, I could make a feature showing the length of time a customer has been with the utility company.
# 
# **Step 6: Choosing a Model**
# I will select appropriate machine learning models for classification. Common options for binary classification tasks like churn prediction include Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machines.
# 
# **Step 7: Training and Evaluating the Model**
# I will divide the dataset into training and testing sets. I will train the chosen models on the training data and evaluate their performance on the testing data using relevant metrics such as accuracy, precision, recall, F1-score, and ROC AUC. I will employ cross-validation for a thorough assessment of the models.
# 
# **Step 8: Testing the Hypothesis**
# To test the client's hypothesis that price sensitivity influences churn, I will do the following:
# - Analyze feature importance to identify which factors have the most impact on churn predictions.
# - Create visual representations of relationships between features like 'cons_12m' and 'churn' to look for patterns.
# - Consider conducting statistical or hypothesis tests to quantify the relationship between price sensitivity features and churn.
# 
# **Step 9: Reporting**
# I will summarize my discoveries in a comprehensive report or email to my AD. I will include insights from the data analysis, model performance, and the results of hypothesis testing. I will offer practical recommendations based on my analysis.

# In[34]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Next, I need to load dataset for 'Client' data.

# In[3]:
file_path = r'C:\Users\Sweet\Downloads\Portofolio\client_data.csv'

# Read the CSV file into a DataFrame
client_data = pd.read_csv(file_path)

# Reset the index and drop the old index
client_data.reset_index(drop=True, inplace=True)

# Display the first few rows of the DataFrame
client_data.head()


# In[4]:
client_data.info()


# In[10]:
# List of date-related features
date_features = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']

# Convert date-related features to datetime and create new year-based features
for f in date_features:
    client_data[f] = pd.to_datetime(client_data[f])

# Create new columns for year-based features
client_data['contract_start_year'] = client_data['date_activ'].dt.year
client_data['contract_end_year'] = client_data['date_end'].dt.year


# In[11]:
print(client_data.isnull().sum())


# In[12]:
def describe_categorical(dataset):
    cat_columns = dataset.select_dtypes(include=['object']).columns.tolist()
    
    if len(cat_columns) != 0:
        print('Categorical variables are', cat_columns, '\n' + '==' * 40)
        for cat in cat_columns:
            value_counts = dataset[cat].value_counts()
            proportion = value_counts / len(dataset)
            describe_frame = pd.DataFrame({'Category': value_counts.index, 'Count': value_counts, 'Proportion': proportion})
            describe_frame.reset_index(drop=True, inplace=True)
            print(f"Description of '{cat}':\n", describe_frame, '\n' + '--' * 40)
    else:
        print('There are no categorical variables in the dataset')


# In[13]:
describe_categorical(client_data)


# In[35]:
data_encoder = LabelEncoder()
client_data['channel_sales'] = data_encoder.fit_transform(client_data['channel_sales'])
origin_encoder = LabelEncoder()
client_data['origin_up'] = origin_encoder.fit_transform(client_data['origin_up'])


# In[36]:
def describe_numeric(dataset):
    # Select numeric columns
    numeric_columns = dataset.select_dtypes(include=['int64', 'float64'])

    if not numeric_columns.empty:
        print('Numeric variables are:', numeric_columns.columns.tolist(), '\n' + '==' * 40)
        
        # Loop through numeric columns and display statistics
        for column in numeric_columns.columns:
            print(f"Column: {column}")
            print(f"Minimum: {dataset[column].min()}")
            print(f"Maximum: {dataset[column].max()}")
            print(f"Mean: {dataset[column].mean()}")
            print(f"Median: {dataset[column].median()}")
            print("--" * 40)
        
    else:
        print('There are no numeric variables in the dataset')


# In[37]:
describe_numeric(client_data)


# In[45]:
# Assuming 'client' of dataset
datetime_columns = client_data.select_dtypes(include=['datetime64[ns]'])

if not datetime_columns.empty:
    # Describe datetime columns with datetime_is_numeric=True
    datetime_description = client_data[datetime_columns.columns].describe(datetime_is_numeric=True)
    
    # Create a new DataFrame to store the result
    datetime_table = pd.DataFrame(datetime_description)
    
    # Rename the index column for better readability
    datetime_table.index.name = 'Statistics'
    
    # Reset the index to make 'Statistics' a regular column
    datetime_table.reset_index(inplace=True)
    
    # Display the result as a pandas DataFrame
    print(datetime_table)
else:
    print('There are no datetime variables in the dataset')


# In[17]:
# Step 1: Problem Formulation
# Define the target variable (churn) and features

X = client_data[['cons_12m', 'imp_cons', 'margin_gross_pow_ele', 'margin_net_pow_ele', 'num_years_antig', 'pow_max']]
y = client_data['churn']

# Step 2: Data Collection and Preprocessing
# Perform data preprocessing steps such as handling missing values and encoding categorical variables
# For simplicity, let's assume there are no missing values and no categorical variables in this example.
# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Model Selection
# Choose a classification model (Random Forest in this example)
clf = RandomForestClassifier(random_state=42)

# Step 7: Model Training and Evaluation
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)


# In[27]:
# Step 4: Exploratory Data Analysis (EDA)
# Visualize the distribution of the target variable 'churn'
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=client_data)
plt.title('Churn Distribution')
plt.show()


# In[19]:
# Define the feature of interest and the target variable
feature_of_interest = 'cons_12m'
target_variable = 'churn'

# Split the data into two groups: churned and not churned customers
churned = client_data[client_data[target_variable] == 1][feature_of_interest]
not_churned = client_data[client_data[target_variable] == 0][feature_of_interest]

# Step 8:
# Hypothesis Testing - t-test
# Perform a t-test to compare means of the 'cons_12m' feature between churned and not churned customers
t_statistic, p_value = ttest_ind(churned, not_churned)

# Display the results
print(f"t-statistic: {t_statistic}")
print(f"P-value: {p_value}")


# In[21]:
# Hypothesis Testing - ANOVA (Analysis of Variance)
# Perform ANOVA to test if 'cons_12m' has a significant impact on churn across multiple categories (e.g., different channel_sales)
anova_result = client_data.groupby('channel_sales')[feature_of_interest].apply(list)
f_statistic, p_value_anova = f_oneway(*anova_result)

# Display the results
print(f"F-statistic (ANOVA): {f_statistic}")
print(f"P-value (ANOVA): {p_value_anova}")


# In[24]:
# Visualize correlations between features and the target variable
correlation_matrix = client_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['churn']], annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation with Churn')
plt.show()


# In[25]:
# Step 5: Feature Engineering (Creating new features, if needed)
# Example: Create a feature indicating the length of the customer's relationship with the utility company
client_data['customer_tenure'] = (pd.to_datetime(client_data['date_end']) - pd.to_datetime(client_data['date_activ'])).dt.days


# In[26]:
client_data['customer_tenure']
