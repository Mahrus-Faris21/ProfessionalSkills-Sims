#!/usr/bin/env python
# coding: utf-8

# # Here is the background information on your task

# The team now has a good understanding of the data and feels confident to use the data to further understand the business problem. The team now needs to brainstorm and build out features to uncover signals in the data that could inform the churn model.
# 
# Feature engineering is one of the keys to unlocking predictive insight through mathematical modeling. Based on the data that is available and was cleaned, identify what you think could be drivers of churn for our client and build those features to later use in your model.
# 
# First focus on building on top of the feature that your colleague has already investigated: **“the difference between off-peak prices in December and January the preceding year”**. After this, if you have time, feel free to get creative with making any other features that you feel are worthwhile.
# 
# Once you have a set of features, you must train a Random Forest classifier to predict customer churn and evaluate the performance of the model with suitable evaluation metrics. Be rigorous with your approach and give full justification for any decisions made by yourself as the intern data scientist. 
# 
# Recall that the hypotheses under consideration is that churn is driven by the customers’ price sensitivities and that it would be possible to predict customers likely to churn using a predictive model.
# 
# If you’re eager to go the extra mile for the client, when you have a trained predictive model, remember to investigate the client’s proposed discounting strategy, with the head of the SME division suggesting that offering customers at high propensity to churn a 20% discount might be effective.
# 
# Build your models and test them while keeping in mind you would need data to prove/disprove the hypotheses, as well as to test the effect of a 20% discount on customers at high propensity to churn.
# 
# 

# # Here is your task

# **Sub-Task 1**
# 
# Your colleague has done some work on engineering the features within the cleaned dataset and has calculated a feature which seems to have predictive power. 
# 
# This feature is **“the difference between off-peak prices in December and January the preceding year”**. 
# 
# Run the cells in the notebook provided (named feature_engineering.ipynb) to re-create this feature. then try to think of ways to improve the feature’s predictive power and elaborate why you made those choices. 
# 
# You should spend 1 - 1.5 hours on this. Be sure to make use of the “feature_engineering.ipynb” notebook to get started with re-creating your colleagues' features.
# 
# **Sub-Task 2**
# 
# Now that you have a dataset of cleaned and engineered features, it is time to build a predictive model to see how well these features are able to predict a customer churning. It is your task to train a Random Forest classifier and to evaluate the results in an appropriate manner. We would also like you to document the advantages and disadvantages of using a Random Forest for this use case. It is up to you how to fulfill this task, but you may want to use the below points to guide your work:
# 
# Ensure you’re able to explain the performance of your model, where did the model underperform?
# Why did you choose the evaluation metrics that you used? Please elaborate on your choices.
# Document the advantages and disadvantages of using the Random Forest for this use case.
# Do you think that the model performance is satisfactory? Give justification for your answer.
# (Bonus) - Relate the model performance to the client's financial performance with the introduction of the discount proposition. How much money could a client save with the use of the model? What assumptions did you make to come to this conclusion?
# You should spend 1 - 1.5 hours on this. When it comes to model evaluation and the explanation of your results, feel free to use the additional links below.
# 
# **If you are stuck:**
# 
# Sub-Task 1
# 
# - Think of ways to evaluate a feature against a label.
# - Think of ways to add new features which would complement the already existing ones. 
# - Think of feature granularity. 
# - Remove unnecessary features.
#  
# Sub-Task 2
# 
# - Is this problem best represented as classification or regression? 
# - What kind of model performance do you think is appropriate? 
# - Most importantly how would you measure such a performance? 
# - How would you tie business metrics such as profits or savings to the model performance?

# # The Answers

# In[76]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind, f_oneway
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


# In[2]:


file_path = r'C:\Users\Sweet\Downloads\Portofolio\client_data.csv'

# Read the CSV file into a DataFrame
client_data = pd.read_csv(file_path)

# Reset the index and drop the old index
client_data.reset_index(drop=True, inplace=True)

# Display the first few rows of the DataFrame
client_data["date_activ"] = pd.to_datetime(client_data["date_activ"], format='%Y-%m-%d')
client_data["date_end"] = pd.to_datetime(client_data["date_end"], format='%Y-%m-%d')
client_data["date_modif_prod"] = pd.to_datetime(client_data["date_modif_prod"], format='%Y-%m-%d')
client_data["date_renewal"] = pd.to_datetime(client_data["date_renewal"], format='%Y-%m-%d')
client_data.head()


# In[3]:


file_path = r'C:\Users\Sweet\Downloads\Portofolio\price_data.csv'

# Read the CSV file into a DataFrame
price_data = pd.read_csv(file_path)

# Reset the index and drop the old index
price_data.reset_index(drop=True, inplace=True)

# Display the first few rows of the DataFrame
price_data["price_date"] = pd.to_datetime(price_data["price_date"], format='%Y-%m-%d')
price_data.head()


# In[4]:


# Group off-peak prices by companies and month
monthly_price_by_id = price_data.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()

# Get january and december prices
jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

# Calculate the difference
diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), jan_prices.drop(columns='price_date'), on='id')
diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
diff = diff[['id', 'offpeak_diff_dec_january_energy','offpeak_diff_dec_january_power']]
diff.head()


# In[24]:


numeric_columns = X.select_dtypes(include=['number']).columns.tolist()

# Handle missing values for numeric columns (impute with the mean)
imputer = SimpleImputer(strategy='mean')
X_numeric_imputed = imputer.fit_transform(X[numeric_columns])

# Continue with data splitting and model training
X_train, X_test, y_train, y_test = train_test_split(X_numeric_imputed, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
class_report = classification_report(y_test, y_pred)

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{class_report}')
print(f'ROC AUC Score: {roc_auc}')


# In[44]:


stat_ = ['max', 'min', 'mean']
price_attr = ['price_off_peak_var', 'price_peak_var', 'price_mid_peak_var', 'price_off_peak_fix', 'price_peak_fix', 'price_mid_peak_fix']

price_stat = (
    price_data.drop(columns=['price_date'])
    .groupby(['id'])
    .agg(**{f'{attr}_{stat}': (attr, stat) for attr in price_attr for stat in stat_})
    .reset_index()
)

price_stat.columns = [''.join(col) if col != 'id' else col for col in price_stat.columns]

price_stat.head(5)


# In[48]:


price_stat = (
    price_stat
    .merge(client_data[['id', 'churn']], on=['id'], how='left')
    .dropna(subset=['churn'])
    .reset_index(drop=True)
)

price_stat.head(5)


# In[49]:


price_stat = (
    price_stat
    .assign(**{f'diff_max_min_{attr}': lambda x: x[f'{attr}_max'] - x[f'{attr}_min'] for attr in price_attr})
)

price_stat.head(7)


# In[53]:


for attr in price_attr:
    price_stat[f'diff_Dec_mean_{attr}'] = price_data[price_data['id'].isin(price_stat['id'])].groupby(['id'])[attr].nth(-1).values - price_stat[f'{attr}_mean']

price_stat.head(7)


# In[71]:


# Filter the price DataFrame for dates after '2015-06-01'
price_filtered = price_data[price_data['price_date'] > '2015-06-01']

# Group by 'id' and calculate the mean for selected columns
price_stat_6_month = price_filtered.groupby('id').agg({
    'price_off_peak_var': ['max', 'min', 'mean'],
    'price_off_peak_fix': ['max', 'min', 'mean']
})

# Flatten the column names
price_stat_6_month.columns = ['_'.join(x) for x in price_stat_6_month.columns]

# Merge with the 'client_data' DataFrame on 'id' to add 'churn' values
price_stat_6_month = price_stat_6_month.merge(client_data[['id', 'churn']], on='id', how='left')

# Drop rows with missing 'churn' values and reset the index
price_stat_6_month.dropna(subset=['churn'], inplace=True)
price_stat_6_month.reset_index(drop=True, inplace=True)

# Calculate and add the 'diff_Dec_mean' columns
for attr in ['price_off_peak_var', 'price_off_peak_fix']:
    price_stat_6_month[f'diff_Dec_mean_{attr}'] = price_data[price_data['id'].isin(price_stat_6_month['id'])].groupby(['id'])[attr].nth(-1).values - price_stat_6_month[f'{attr}_mean']

# Display the first 7 rows of the resulting dataframe
price_stat_6_month.head(7)


# In[72]:


# Filter the price DataFrame for dates after '2015-09-01'
price_filtered = price_data[price_data['price_date'] > '2015-09-01']

# Group by 'id' and calculate the mean for selected columns
price_stat_3_month = price_filtered.groupby('id').agg({
    'price_off_peak_var': ['mean'],
    'price_off_peak_fix': ['mean'],
})

# Flatten the column names
price_stat_3_month.columns = ['_'.join(x) for x in price_stat_3_month.columns]

# Merge with the 'client_data' DataFrame on 'id' to add 'churn' values
price_stat_3_month = price_stat_3_month.merge(client_data[['id', 'churn']], on='id', how='left')

# Drop rows with missing 'churn' values and reset the index
price_stat_3_month.dropna(subset=['churn'], inplace=True)
price_stat_3_month.reset_index(drop=True, inplace=True)

# Calculate and add the 'diff_Dec_mean' columns
for attr in ['price_off_peak_var', 'price_off_peak_fix']:
    price_stat_3_month[f'diff_Dec_mean_{attr}'] = price_data[price_data['id'].isin(price_stat_3_month['id'])].groupby(['id'])[attr].nth(-1).values - price_stat_3_month[f'{attr}_mean']

# Display the first 7 rows of the resulting dataframe
price_stat_3_month.head(7)


# In[74]:


# Prepare the data and drop unnecessary columns
train_data = client_data.copy()
train_data['year_modif_prod'] = train_data['date_modif_prod'].dt.year
train_data['year_renewal'] = train_data['date_renewal'].dt.year
train_data = train_data.drop(columns=['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'])

# Encode 'has_gas' using LabelEncoder
has_gas_encoder = LabelEncoder()
train_data['has_gas'] = has_gas_encoder.fit_transform(train_data['has_gas'])

# Define a function to calculate the difference between the last and first prices for a given attribute
def calculate_price_difference(df, attribute):
    return df.groupby('id')[attribute].last() - df.groupby('id')[attribute].first()

# Calculate price differences for 'price_off_peak_var' and 'price_off_peak_fix'
diff_dec_jan_off_peak_var = calculate_price_difference(price_data.sort_values(by=['price_date']), 'price_off_peak_var')
diff_dec_jan_off_peak_fix = calculate_price_difference(price_data.sort_values(by=['price_date']), 'price_off_peak_fix')

# Reset index and rename columns
diff_dec_jan_off_peak_var = diff_dec_jan_off_peak_var.reset_index(name='diff_dec_jan_off_peak_var')
diff_dec_jan_off_peak_fix = diff_dec_jan_off_peak_fix.reset_index(name='diff_dec_jan_off_peak_fix')

# Merge the calculated differences with train_data
train_data = train_data.merge(diff_dec_jan_off_peak_var, on='id', how='left')
train_data = train_data.merge(diff_dec_jan_off_peak_fix, on='id', how='left')

# Calculate and add differences for other price attributes
price_attributes = ['price_peak_var', 'price_peak_fix', 'price_mid_peak_var', 'price_mid_peak_fix']
for attr in price_attributes:
    diff_dec_jan_temp = calculate_price_difference(price.sort_values(by=['price_date']), attr)
    diff_dec_jan_temp = diff_dec_jan_temp.reset_index(name=f'diff_dec_jan_{attr}')
    train_data = train_data.merge(diff_dec_jan_temp, on='id', how='left')

# Merge with price_stat for price changing trends
price_stat_attributes = [
    'diff_Dec_mean_price_off_peak_var',
    'diff_Dec_mean_price_off_peak_fix',
    'diff_Dec_mean_price_peak_var',
    'diff_Dec_mean_price_peak_fix',
    'diff_Dec_mean_price_mid_peak_var',
    'diff_Dec_mean_price_mid_peak_fix'
]
train_data = train_data.merge(price_stat[['id'] + price_stat_attributes], on='id', how='left')

# Display the first few rows of the resulting DataFrame
train_data.head()


# In[75]:


# Drop columns 'id' and 'churn' to create the feature matrix X and target vector y
X = train_data.drop(columns=['id', 'churn'])
y = train_data['churn']

# Check the shape of X and y
X.shape, y.shape


# In[77]:


def train_and_evaluate_rf(X, y, random_state=56, n_splits=5, shuffle=True):
    # Create an array to store predicted training labels
    pred_train_labels = np.zeros(shape=(X.shape[0], 2))

    # Create a DataFrame to store feature importances
    feature_importance_df = pd.DataFrame(data={'feature_name': X.columns, 'feature_importance': [0] * len(X.columns)})

    # Create stratified k-fold cross-validation datasets
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    fold_counter = 1

    # Iterate over folds
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Build and train the Random Forest model
        rf = RandomForestClassifier(random_state=random_state)
        rf.fit(X_train, y_train)

        # Predict probabilities for the test set
        pred_proba = rf.predict_proba(X_test)
        pred_train_labels[test_index] = pred_proba

        # Update feature importances
        feature_importance_df['feature_importance'] += rf.feature_importances_

        # Calculate and print metrics for the fold
        precision = precision_score(y_test, rf.predict(X_test))
        recall = recall_score(y_test, rf.predict(X_test))
        accuracy = accuracy_score(y_test, rf.predict(X_test))
        print(f"Fold {fold_counter} Precision {precision:.3f} Recall {recall:.3f} Accuracy {accuracy:.3f}")
        fold_counter += 1

    # Predicted labels
    pred_y = pred_train_labels.argmax(axis=-1)

    # Calculate and return overall metrics
    total_precision = precision_score(y, pred_y)
    total_recall = recall_score(y, pred_y)
    total_accuracy = accuracy_score(y, pred_y)
    
    return feature_importance_df, total_precision, total_recall, total_accuracy

# Call the function to train and evaluate the Random Forest model
feature_importance_df, total_precision, total_recall, total_accuracy = train_and_evaluate_rf(X, y)

# Print overall metrics
print(f"Total Precision {total_precision:.3f} Recall {total_recall:.3f} Accuracy {total_accuracy:.3f}")


# In[80]:


def plot_feature_importance(feature_importance_df, figsize=(12, 10), palette='winter'):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create a barplot of feature importances with custom palette
    sns.barplot(data=feature_importance_df.sort_values(by=['feature_importance'], ascending=False), 
                y='feature_name', 
                x='feature_importance', 
                ax=ax, 
                palette=palette)

    # Set labels and title
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature Name')
    ax.set_title('Feature Importance Plot')

    # Show the plot
    plt.show()

# Call the function to plot feature importances with a custom color palette
plot_feature_importance(feature_importance_df, palette='winter')


# In[92]:


# Predicted labels
pred_train_labels = np.zeros(shape=(X.shape[0], 2))
pred_y = pred_train_labels.argmax(axis=-1)

# Calculate and print overall metrics
total_precision = precision_score(y, pred_y)
total_recall = recall_score(y, pred_y)
total_accuracy = accuracy_score(y, pred_y)

# Print classification report
print(classification_report(y, pred_y))


# In[91]:


# Calculate the confusion matrix
CM = confusion_matrix(y, pred_y)

# Obtain unique class labels
class_labels = np.unique(y)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(CM, annot=True, fmt="d", cmap="Blues", cbar=True, 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[103]:


def engineer_features(data, price_data, discount=0.8):
    # Drop unnecessary columns
    data = data.drop(columns=['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'])

    # Encode 'has_gas' column
    has_gas_encoder = LabelEncoder()
    data['has_gas'] = has_gas_encoder.fit_transform(data['has_gas'])

    # Calculate and add the 'diff_Dec_mean' columns
    for attr in ['price_off_peak_var', 'price_off_peak_fix']:
        data[f'diff_Dec_mean_{attr}'] = (
            data['id'].map(price_data.set_index('id')[f'diff_Dec_mean_{attr}']) * discount
        )

    return data

# Call the feature engineering function for test_data
test_data = engineer_features(client_data.copy(), price_stat_3_month)

# Display the resulting test_data
test_data.head()


# In[107]:


# Initialize arrays to store results for each fold
precisions, recalls, accuracies = [], [], []
prob_no_discount_list, prob_discount_list = [], []

# Create cross-validation dataset
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=29)

for fold_counter, (train_index, test_index) in enumerate(kfold.split(X, y), start=1):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Build and train the model
    rf = RandomForestClassifier(random_state=56)
    rf.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate precision, recall, and accuracy for this fold
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Fold {fold_counter} Precision {precision:.3f} Recall {recall:.3f} Accuracy {accuracy:.3f}")

    # Append fold results to lists
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)

    # Predict probabilities for the test set
    prob_test = rf.predict_proba(X_test)
    prob_discount_list.append(prob_test)


# In[108]:


# Calculate and print the mean values across folds
mean_precision = sum(precisions) / len(precisions)
mean_recall = sum(recalls) / len(recalls)
mean_accuracy = sum(accuracies) / len(accuracies)

print(f"Mean Precision {mean_precision:.3f} Mean Recall {mean_recall:.3f} Mean Accuracy {mean_accuracy:.3f}")

