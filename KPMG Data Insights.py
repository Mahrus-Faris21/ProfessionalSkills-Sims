#!/usr/bin/env python
# coding: utf-8

# **Please make sure you attempt Module 1 before starting this Module**
# 
# Sprocket Central Pty Ltd has given us a new list of 1000 potential customers with their demographics and attributes. However, these customers do not have prior transaction history with the organisation. 
# 
# The marketing team at Sprocket Central Pty Ltd is sure that, if correctly analysed, the data would reveal useful customer insights which could help optimise resource allocation for targeted marketing. Hence, improve performance by focusing on high value customers.

# # Here is your task

# For context, Sprocket Central Pty Ltd is a long-standing KPMG client whom specialises in high-quality bikes and accessible cycling accessories to riders. Their marketing team is looking to boost business by analysing their existing customer dataset to determine customer trends and behaviour. 
# 
# Using the existing 3 datasets (Customer demographic, customer address and transactions) as a labelled dataset, please recommend which of these 1000 new customers should be targeted to drive the most value for the organisation. 
# 
# In building this recommendation, we need to start with a PowerPoint presentation which outlines the approach which we will be taking. The client has agreed on a 3 week scope with the following 3 phases as follows - Data Exploration; Model Development and Interpretation.
# 
# Prepare a detailed approach for completing the analysis including activities – i.e. understanding the data distributions, feature engineering, data transformations, modelling, results interpretation and reporting. This detailed plan needs to be presented to the client to get a sign-off. Please advise what steps you would take. 
# 
# 
# Please ensure your PowerPoint presentation includes a detailed approach for our strategy behind each of the 3 phases including activities involved in each - i.e. understanding the data distributions, feature engineering, data transformations, modelling, results interpretation and reporting. This detailed plan needs to be presented to the client to get a sign-off.
# 
# --
# 
# Tips: Raw data fields may be transformed into other calculated fields for modelling purposes (i.e. converting D.O.B to age or age groups).  Tips: You may source external data from the ABS / Census to add additional variables that may help support your model. 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns


# In[2]:


sheet_name_to_read = 'NewCustomerList'
file_path = r'C:\Users\Sweet\Downloads\Portofolio\KPMG_VI_New_raw_data_update_final.xlsx'

SprocketCentral = pd.read_excel(file_path, sheet_name=sheet_name_to_read)
new_header = SprocketCentral.iloc[0]
SprocketCentral = SprocketCentral[1:]
SprocketCentral.columns = new_header
SprocketCentral['property_valuation'] = SprocketCentral['property_valuation'].astype(str)
SprocketCentral['DOB'] = pd.to_datetime(SprocketCentral['DOB'])

SprocketCentral.reset_index(drop=True, inplace=True)
SprocketCentral.head()


# In[3]:


# Explore the data
SprocketCentral[['past_3_years_bike_related_purchases', 'property_valuation', 'Rank', 'Value']].info()


# In[4]:


# Calculate summary statistics
SprocketCentral[['past_3_years_bike_related_purchases', 'property_valuation', 'Rank', 'Value']].describe().transpose()


# In[5]:


# Identify missing values
print(SprocketCentral[['past_3_years_bike_related_purchases', 'property_valuation', 'Rank', 'Value']].isnull().sum())


# In[34]:


# Sample data
SprocketCentral['past_3_years_bike_related_purchases'] = SprocketCentral['past_3_years_bike_related_purchases'].astype(str)
SprocketCentral['past_3_years_bike_related_purchases'] = SprocketCentral['past_3_years_bike_related_purchases'].str.replace('[^0-9]', '', regex=True)
SprocketCentral['property_valuation'] = pd.to_numeric(SprocketCentral['property_valuation'], errors='coerce')
SprocketCentral['past_3_years_bike_related_purchases'] = pd.to_numeric(SprocketCentral['past_3_years_bike_related_purchases'], errors='coerce')

bike_purchases = SprocketCentral['past_3_years_bike_related_purchases']
property_valuation = SprocketCentral['property_valuation']
rank = SprocketCentral['Rank']
value = SprocketCentral['Value']

# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(12, 4))

# Define custom tick positions and labels based on data range
custom_ticks = [
    np.arange(0, bike_purchases.max(), 10),  # Custom ticks for 'Bike Purchases'
    np.arange(property_valuation.min(), property_valuation.max() + 1),  # Custom ticks for 'Property Valuation'
    np.arange(rank.min(), rank.max() + 1, 1000),  # Custom ticks for 'Rank'
    np.arange(value.min(), value.max()),  # Custom ticks for 'Value'
]

# Create histograms and set custom ticks and labels
for i, (data, title, ticks) in enumerate(zip(
    [bike_purchases, property_valuation, rank, value],
    ['Bike Purchases', 'Property Valuation', 'Rank', 'Value'],
    custom_ticks
)):
    axes[i].hist(data.dropna(), bins=20, color='skyblue', edgecolor='black')  # Drop NaN values
    axes[i].set_xlabel(title)
    axes[i].set_ylabel('Count')
    axes[i].set_xticks(ticks)
    if title not in ['Rank', 'Value']:
        axes[i].set_xticklabels([str(int(tick)) for tick in ticks])

plt.tight_layout()
plt.show()


# In[117]:


# Define the list of categorical columns to visualize
categorical_columns = ['job_industry_category', 'wealth_segment', 'state']

# Create subplots for each column
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
axes = axes.flatten()  # Flatten the axes array for easier indexing

for i, column in enumerate(categorical_columns):
    most_common_value = SprocketCentral[column].mode().values[0]
    
    # Create a count plot (bar plot) for distribution
    sns.countplot(data=SprocketCentral, x=column, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')
    
    # Display the most common value in the title
    axes[i].set_title(f'Distribution of {column}\n(Most Common: {most_common_value})')
    
    if column == 'job_industry_category':
        axes[i].tick_params(axis='x', rotation=90)

    for bar in axes[i].patches:
        height = bar.get_height()
        axes[i].annotate(f'{int(height)}', (bar.get_x() + bar.get_width() / 2, height),
                         ha='center', va='bottom', fontsize=10, color='black')
    
# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# In[125]:


specific_job_titles = [
    'Business Systems Development Analyst',
    'Tax Accountant',
    'Social Worker',
    'Recruiting Manager',
    'Internal Auditor',
    'Administrative Assistant II ',
    'Health Coach III',
    'Health Coach I',
    'Research Assistant III'
]

filtered_data = SprocketCentral[SprocketCentral['job_title'].isin(specific_job_titles)]

plt.figure(figsize=(6, 6))
job_title_distribution = filtered_data['job_title'].value_counts()
labels = job_title_distribution.index
sizes = job_title_distribution.values
colors = sns.color_palette('viridis', len(labels))

patches, texts, autotexts = plt.pie(
    sizes, labels=labels, colors=colors, autopct='%1.1f%%',
    shadow=True, startangle=140, pctdistance=0.85
)
plt.axis('equal')

for autotext in autotexts:
    autotext.set_color('white')

plt.show()


# In[46]:


# Feature Engineering: Create 'bicycle property' and 'customer_rank' feature
def bicycle_group(purchases, valuation):
    if purchases < 50 and valuation <= 7:
        return 'Low Activity, Low Valuation'
    elif purchases >= 50 and valuation > 7:
        return 'High Activity, High Valuation'
    else:
        return 'Other'

def customer_rank(rank_val):
    if rank_val < 50:
        return 'Low Rank'
    elif rank_val >= 50:
        return 'High Rank'
    else:
        return 'Other'


# In[47]:


SprocketCentral['past_3_years_bike_related_purchases'] = pd.to_numeric(SprocketCentral['past_3_years_bike_related_purchases'])
SprocketCentral['property_valuation'] = pd.to_numeric(SprocketCentral['property_valuation'])

SprocketCentral['bicycle property'] = SprocketCentral.apply(lambda row: bicycle_group(row['past_3_years_bike_related_purchases'], row['property_valuation']), axis=1)
SprocketCentral['customer ranking'] = SprocketCentral.apply(lambda row: customer_rank(row['Rank']), axis=1)


# In[27]:


SprocketCentral['bicycle property'].value_counts()


# In[48]:


SprocketCentral['customer ranking'].value_counts()


# In[49]:


import statsmodels.api as sm


# In[50]:


bikePurchases = np.array(SprocketCentral['past_3_years_bike_related_purchases'])
propertyValue = np.array(SprocketCentral['property_valuation'])
ranK = np.array(SprocketCentral['Rank'])
valuE = np.array(SprocketCentral['Value'])


# In[63]:


# Combine NumPy arrays into a single NumPy array for X
X = np.column_stack((bikePurchases, propertyValue, ranK))

# Use the 'Value' column as the target variable (y)
y = valuE

# Split the data into training and testing sets & Identify non-numeric values and handle them
non_numeric_mask = np.logical_not(np.isreal(X_train))
non_numeric_values = X_train[non_numeric_mask]
X_train[non_numeric_mask] = np.nan  # Replace non-numeric values with NaN or another appropriate value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype(np.float64)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[65]:


missing_values = np.isnan(X_train)
missing_values


# In[73]:


bikePurchases = bikePurchases.astype(np.float64)
propertyValue = propertyValue.astype(np.float64)
ranK = ranK.astype(np.float64)
valuE = valuE.astype(np.float64)

X = np.column_stack((bikePurchases, propertyValue, ranK))
X = np.column_stack((np.ones(X.shape[0]), X))
y = valuE

coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

# Make predictions
y_pred = X @ coefficients


# In[75]:


# Print the coefficients and predictions
print("Coefficients:", coefficients)


# In[79]:


print("Predicted Values:", y_pred[:5])


# In[80]:


# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print a summary of the regression results
print(model.summary())


# In[88]:


plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)  # Scatter plot of observed vs. predicted values
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line for perfect prediction
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Observed vs. Predicted Values')

annotation_text = 'X: Variables Bike Purchases, Property Value, Rank Customer\nY: Value Customer'
box_properties = dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='yellow')
box_properties['pad'] = 50
plt.text(0.1, 0.9, annotation_text, transform=plt.gca().transAxes, bbox=box_properties)

plt.show()


# In[94]:


data = pd.DataFrame({
    'bikePurchases': bikePurchases,
    'propertyValue': propertyValue,
    'ranK': ranK,
    'valuE': valuE
})

correlation_matrix = data.corr()

plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='YlGnBu',
    linewidths=.5,
    annot_kws={"va": "center"}  # Set vertical alignment to "center"
)

ax.xaxis.set_ticks_position('top')
plt.show()

