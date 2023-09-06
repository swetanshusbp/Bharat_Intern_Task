#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# In[2]:


# Set the aesthetic style of the plots
sns.set_style("whitegrid")


# In[3]:


# Load dataset
wine = pd.read_csv("C:\\Users\\sweta\\Downloads\\archive (5)\\winequality-red.csv")


# In[4]:


# Display initial data
print(wine.head())
print(wine.info())


# In[5]:


# Plot various features against wine quality
features_to_plot = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='quality', y=feature, data=wine)
    plt.title(f'{feature} vs. Wine Quality')
    plt.show()


# In[6]:


# Convert wine quality into binary labels
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)


# In[7]:


# Encode labels
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[8]:


# Show quality distribution
print(wine['quality'].value_counts())
sns.countplot(wine['quality'])
plt.show()


# In[9]:


# Split data into train and test sets
X = wine.drop('quality', axis=1)
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Standardize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  # Use transform instead of fit_transform for test set


# In[11]:


# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print("Random Forest Classifier Report:\n", classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# In[12]:


# SGD Classifier
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print("SGD Classifier Report:\n", classification_report(y_test, pred_sgd))
print(confusion_matrix(y_test, pred_sgd))


# In[13]:


# Support Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print("Support Vector Classifier Report:\n", classification_report(y_test, pred_svc))


# In[14]:


# Grid Search for SVC optimization
param_grid = {
    'C': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param_grid, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)


# In[15]:


# Best parameters and retraining SVC
print("Best SVC Parameters:", grid_svc.best_params_)
svc_optimized = SVC(**grid_svc.best_params_)
svc_optimized.fit(X_train, y_train)
pred_svc_optimized = svc_optimized.predict(X_test)
print("Optimized SVC Report:\n", classification_report(y_test, pred_svc_optimized))


# In[16]:


# Cross-validation for Random Forest
rfc_eval = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10)
print(f"Random Forest Cross Val Score: {rfc_eval.mean()}")


# In[17]:


# Import Plotly for 3D plotting
import plotly.express as px
import numpy as np

# 3D Scatter Plot using Plotly
fig = px.scatter_3d(wine, x='fixed acidity', y='volatile acidity', z='citric acid',
                    color='quality', color_continuous_scale='Viridis')
fig.show()


# In[30]:


# Heatmap for Correlation
plt.figure(figsize=(12, 6))
sns.heatmap(wine.corr(), annot=True, cmap='RdBu_r')
plt.title("Feature Correlation Heatmap")
plt.show()


# In[19]:


# Model Comparisons
# Collect the scores for different models
model_names = ['Random Forest', 'SGD', 'SVC', 'Optimized SVC']
model_scores = [
    classification_report(y_test, pred_rfc, output_dict=True)['accuracy'],
    classification_report(y_test, pred_sgd, output_dict=True)['accuracy'],
    classification_report(y_test, pred_svc, output_dict=True)['accuracy'],
    classification_report(y_test, pred_svc_optimized, output_dict=True)['accuracy']
]


# In[20]:


# Bar graph for model comparisons
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=model_scores)
plt.title('Model Comparison by Accuracy')
plt.ylabel('Accuracy')
plt.show()


# In[21]:


# Random Forest Cross-validation Score
print(f"Random Forest Cross Val Score: {rfc_eval.mean()}")


# In[ ]:




