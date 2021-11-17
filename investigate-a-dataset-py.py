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
# > Some questions that can be answered based on the Profit of movies making more then 25M Dollars:
# 1. What is the average budget of the movie?
# 2. What is the average revenue of the movie?
# 3. What is the average runtime of the movie?
# 4. Which are the successfull genres?
# 5. Which are the most frequent cast involved?

# In[40]:


import numpy as np
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# In[5]:


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

# In[6]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
tmdb_data.head()


# In[7]:


tmdb_data.describe()


# In[8]:


tmdb_data.info()


# In[9]:


sum(tmdb_data.duplicated())


# In[10]:


# Columns that needs to be deleted
deleted_columns = [ 'id', 'imdb_id', 'popularity', 'budget_adj', 'revenue_adj', 'homepage', 'keywords', 'director', 'tagline', 'overview', 'production_companies', 'vote_count', 'vote_average']
# Drop the columns from the database
tmdb_data.drop(deleted_columns, axis=1, inplace=True)
# Lets look at the new dataset
tmdb_data.head()


# In[11]:


# Store rows and columns using shape function.
rows, col = tmdb_data.shape
#since rows includes count of a header, we need to remove its count.
print('We have {} total rows and {} columns.'.format(rows-1, col))


# In[12]:


# Drop duplicate rows but keep the first one
tmdb_data.drop_duplicates(keep = 'first', inplace = True)
# Store rows and columns using shape function.
rows, col = tmdb_data.shape
print('Now we have {} total rows and {} columns.'.format(rows-1, col))


# In[13]:


# Columns that need to be checked.
columns = ['budget', 'revenue']
# Replace 0 with NAN
tmdb_data[columns] = tmdb_data[columns].replace(0, np.NaN)
# Drop rows which contains NAN
tmdb_data.dropna(subset = columns, inplace = True)
rows, col = tmdb_data.shape
print('We now have only {} rows.'.format(rows-1))


# In[14]:


# Convert column release_date to DateTime
tmdb_data.release_date = pd.to_datetime(tmdb_data['release_date'])
# Lets look at the new dataset
tmdb_data.head()


# In[15]:


# Columns to convert datatype of
columns = ['budget', 'revenue']
# Convert budget and revenue column to int datatype
tmdb_data[columns] = tmdb_data[columns].applymap(np.int64)
# Lets look at the new datatype
tmdb_data.dtypes


# In[16]:


# Replace runtime value of 0 to NAN, Since it will affect the result.
tmdb_data['runtime'] = tmdb_data['runtime'].replace(0, np.NaN)
# Check the stats of dataset
tmdb_data.describe()


# # Explor

# ## Research Question 1.1 (Which movie had the highest and lowest profit?)

# In[17]:


# To calculate profit, we need to substract the budget from the revenue.
tmdb_data['profit'] = tmdb_data['revenue'] - tmdb_data['budget']
# Lets look at the new dataset
tmdb_data.head()


# In[18]:


# Movie with highest profit
tmdb_data.loc[tmdb_data['profit'].idxmax()]


# In[19]:


# Movie with lowest profit
tmdb_data.loc[tmdb_data['profit'].idxmin()]


# ## Research Question 1.2 (Which movie had the greatest and least runtime?)

# In[21]:


# Movie with greatest runtime
tmdb_data.loc[tmdb_data['runtime'].idxmax()]


# In[22]:


# Movie with least runtime
tmdb_data.loc[tmdb_data['runtime'].idxmin()]


# ## Research Question 1.3 (What is the average runtime of all movies?)

# In[23]:


# Average runtime of movies
tmdb_data['runtime'].mean()


# In[24]:


# x-axis
plt.xlabel('Runtime of Movies in Minutes')
# y-axis
plt.ylabel('Number of Movies')
# Title of the histogram
plt.title('Runtime distribution of all the movies')
# Plot a histogram
plt.hist(tmdb_data['runtime'], bins = 50)


# In[25]:


# x-axis
plt.xlabel('Runtime in Minutes')
# y-axis
plt.ylabel('Profit in Dollars')
# Title of the histogram
plt.title('Relationship between runtime and profit')
plt.scatter(tmdb_data['runtime'], tmdb_data['profit'], alpha=0.5)
plt.show()


# ## Research Question 1.4 (Which movie had the highest and lowest budget?)

# In[26]:


# Movie with highest budget
tmdb_data.loc[tmdb_data['budget'].idxmax()]


# In[27]:


# Movie with lowest budget
tmdb_data.loc[tmdb_data['budget'].idxmin()]


# In[28]:


# x-axis
plt.xlabel('Budget in Dollars')
# y-axis
plt.ylabel('Profit in Dollars')
# Title of the histogram
plt.title('Relationship between budget and profit')
plt.scatter(tmdb_data['budget'], tmdb_data['profit'], alpha=0.5)
plt.show()


# ## Research Question 1.5 (Which movie had the highest and lowest revenue?)

# In[29]:


# Movie with highest revenue
tmdb_data.loc[tmdb_data['revenue'].idxmax()]


# In[30]:


# Movie with lowest revenue
tmdb_data.loc[tmdb_data['revenue'].idxmin()]


# In[31]:


# x-axis
plt.xlabel('Revenue in Dollars')
# y-axis
plt.ylabel('Profit in Dollars')
# Title of the histogram
plt.title('Relationship between revenue and profit')
plt.scatter(tmdb_data['revenue'], tmdb_data['profit'], alpha=0.5)
plt.show()


# In[32]:


# x-axis
plt.xlabel('Revenue in Dollars')
# y-axis
plt.ylabel('Budget in Dollars')
# Title of the histogram
plt.title('Relationship between revenue and budget')
plt.scatter(tmdb_data['revenue'], tmdb_data['budget'], alpha=0.5)
plt.show()


# ## Research Question 2.1 (What is the average budget of the movie w.r.t Profit of movies making more then 25M Dollars?)

# In[33]:


# Dataframe which has data of movies which made profit of more the 25M Dollars.
tmdb_profit_data = tmdb_data[tmdb_data['profit'] >= 25000000]
# Reindexing the dataframe
tmdb_profit_data.index = range(len(tmdb_profit_data))
#showing the dataset
tmdb_profit_data.head()


# In[34]:


# Printing the info of the new dataframe
tmdb_profit_data.info()


# In[35]:


# Finfd the average budget of movies which made profit more then 25M Dollars
tmdb_profit_data['budget'].mean()


# ## Research Question 2.2 (What is the average revenue of the movie w.r.t Profit of movies making more then 25M Dollars?)

# In[36]:


# Finfd the average revenue of movies which made profit more then 25M Dollars
tmdb_profit_data['revenue'].mean()


# ## Research Question 2.3 (What is the average runtime of the movie w.r.t Profit of movies making more then 25M Dollars?)

# In[ ]:


tmdb_profit_data['runtime'].mean()


# ## Research Question 2.4 (Which are the successfull genres w.r.t Profit of movies making more then 25M Dollars?)

# In[37]:


# This will first concat all the data with | from the whole column and then split it using | and count the number of times it occured. 
genres_count = pd.Series(tmdb_profit_data['genres'].str.cat(sep = '|').split('|')).value_counts(ascending = False)
genres_count


# In[38]:


# Initialize the plot
diagram = genres_count.plot.bar(fontsize = 8)
# Set a title
diagram.set(title = 'Top Genres')
# x-label and y-label
diagram.set_xlabel('Type of genres')
diagram.set_ylabel('Number of Movies')
# Show the plot
plt.show()


# ## Research Question 2.5 (Which are the most frequent cast involved w.r.t Profit of movies making more then 25M Dollars?)

# In[41]:


# This will first concat all the data with | from the whole column and then split it using | and count the number of times it occured. 
cast_count = pd.Series(tmdb_profit_data['cast'].str.cat(sep = '|').split('|')).value_counts(ascending = False)
cast_count.head(20)


# In[42]:


# Initialize the plot
diagram = cast_count.head(20).plot.barh(fontsize = 8)
# Set a title
diagram.set(title = 'Top Cast')
# x-label and y-label
diagram.set_xlabel('Number of Movies')
diagram.set_ylabel('List of cast')
# Show the plot
plt.show()


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

# In[43]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




