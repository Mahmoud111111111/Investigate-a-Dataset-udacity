#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset (TMDb Database)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# > In this section of the report, I'll provide a brief introduction to the dataset I've selected for analysis. At the end of this section, I wil describe the questions that I plan on exploring over the course of the report.
# ### Dataset Description 
# 
# > I will be using TMDB movie dataset, This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue.
# 
# ### Question(s) for Analysis
#    > Some general questions that can be answered are:
# 1. Which movie had the highest and lowest profit?
# 2. Which movie had the greatest and least runtime?
# 3. What is the average runtime of all movies?
# 4. Which movie had the highest and lowest budget?
# 5. Which movie had the highest and lowest revenue?
# 

# In[1]:


import numpy as np
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# In[3]:


tmdb_data = pd.read_csv('tmdb-movies.csv')
tmdb_data.head()


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > In this section of the report, I will check for cleanliness, and then trim and clean my dataset for analysis.
# ### Observations from above dataset are:
#        1. The dataset has not provided the currency for columns we will be dealing with hence we will assume it is in dollars.
#        1. Even the vote count is not same for all the movies and hence this affects the vote average column.
# 
# ### General Properties
# > Let's check the dataset and see what cleaning does it requires.

# In[4]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
tmdb_data.head()


# In[5]:


tmdb_data.describe()


# In[6]:


tmdb_data.info()


# In[7]:


sum(tmdb_data.duplicated())


# In[8]:


deleted_columns = [ 'id', 'imdb_id', 'popularity', 'budget_adj', 'revenue_adj', 'homepage', 'keywords', 'director', 'tagline', 'overview', 'production_companies', 'vote_count', 'vote_average']
tmdb_data.drop(deleted_columns, axis=1, inplace=True)
tmdb_data.head()


# In[9]:


rows, col = tmdb_data.shape
print('We have {} total rows and {} columns.'.format(rows-1, col))


# In[10]:


tmdb_data.drop_duplicates(keep = 'first', inplace = True)
rows, col = tmdb_data.shape
print('our rows {} and our {} columns.'.format(rows-1, col))


# In[11]:


columns = ['budget', 'revenue']
tmdb_data[columns] = tmdb_data[columns].replace(0, np.NaN)
tmdb_data.dropna(subset = columns, inplace = True)
rows, col = tmdb_data.shape
print('We have {} rows.'.format(rows-1))


# In[12]:


tmdb_data.release_date = pd.to_datetime(tmdb_data['release_date'])
tmdb_data.head()


# In[13]:


columns = ['budget', 'revenue']
tmdb_data[columns] = tmdb_data[columns].applymap(np.int64)
tmdb_data.dtypes


# In[14]:


tmdb_data['runtime'] = tmdb_data['runtime'].replace(0, np.NaN)
tmdb_data.describe()


# # Explor

# ## Research Question 1.1 (Which movie had the highest and lowest profit?)

# In[15]:


tmdb_data['profit'] = tmdb_data['revenue'] - tmdb_data['budget']
tmdb_data.head()


# In[16]:


tmdb_data.loc[tmdb_data['profit'].idxmax()]


# In[17]:


tmdb_data.loc[tmdb_data['profit'].idxmin()]


# ## Research Question 1.2 (Which movie had the greatest and least runtime?)

# In[18]:


tmdb_data.loc[tmdb_data['runtime'].idxmax()]


# In[19]:


tmdb_data.loc[tmdb_data['runtime'].idxmin()]


# ## Research Question 1.3 (What is the average runtime of all movies?)

# In[20]:


tmdb_data['runtime'].mean()


# In[25]:


plt.title('Relationship between profit & runtime ')
plt.ylabel('Profit in USD')
plt.xlabel('Runtime in Minutes')
plt.scatter(tmdb_data['runtime'], tmdb_data['profit'], alpha=0.5);


# In[23]:


plt.title('Runtime distribution of all ')
plt.ylabel('Count of Movies')
plt.xlabel('Runtime in Minutes')
plt.hist(tmdb_data['runtime'], bins = 50);


# ## Research Question 1.4 (Which movie had the highest and lowest budget?)

# In[26]:


tmdb_data.loc[tmdb_data['budget'].idxmin()]


# In[27]:


tmdb_data.loc[tmdb_data['budget'].idxmax()]


# In[29]:


plt.title('Relationship between profit & budget')
plt.ylabel('Profit in USD')
plt.xlabel('Budget in USD')
plt.scatter(tmdb_data['budget'], tmdb_data['profit'], alpha=0.5);


# ## Research Question 1.5 (Which movie had the highest and lowest revenue?)

# In[30]:


tmdb_data.loc[tmdb_data['revenue'].idxmin()]


# In[31]:


tmdb_data.loc[tmdb_data['revenue'].idxmax()]


# In[33]:


plt.title('Relationship between budget & revenue ')
plt.ylabel('Budget in USD')
plt.xlabel('Revenue in USD')
plt.scatter(tmdb_data['revenue'], tmdb_data['budget'], alpha=0.5);


# In[35]:


plt.title('Relationship between profit & revenue ')
plt.ylabel('Profit in USD')
plt.xlabel('Revenue in USD')
plt.scatter(tmdb_data['revenue'], tmdb_data['profit'], alpha=0.5);


# # Conclusions
# 
# > So the conclusion is, that if we want to create movies which can give us a profit of more then 25M Dollars then
# The average budget of the movies can be arround 51870307.75 Dollars
# The average runtime of the movies can be arround 112.56 Minutes
# The Top 10 Genres we should focus on should be Drama, Comedy, Action, Thriller, Adventure, Romance, Crime, Family, Scince Fiction, Fantasy
# The Top 5 cast we should focus on should be Tom Cruise, Tom Hanks, Brad Pitt, Robert De Niro, Bruce Willis
# The average revenue of the movies will be arround 206359440.87 Dollars
# 
# ### The limitations associated with the conclusions are:
# 
# > The conclusion is not full proof that given the above requirement the movie will be a big hit but it can be.
# Also, we also lost some of the data in the data cleaning steps where we dont know the revenue and budget of the movie, which has affected our analysis.
# This conclusion is not error proof.

# In[39]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# 
