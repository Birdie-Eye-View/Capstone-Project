# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. Acquire and Prepare
4. Exploration
5. Modeling
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions in order to expedite and maintain cleanliness
of the final_report.ipynb
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import pandas as pd
import snscrape.modules.twitter as sntwitter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import re
import textblob
from textblob import TextBlob

from wordcloud import WordCloud, STOPWORDS
from emot.emo_unicode import UNICODE_EMOJI
lemmatizer = WordNetLemmatizer()

from wordcloud import ImageColorGenerator
from PIL import Image
import warnings
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import csv

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =======================================================================================================
# Imports END
# Acquire and Prepare START
# =======================================================================================================


def acquire_data():
    '''
    acquire_data reads the data .csv file and returns 
    a pandas dataframe.
    '''
    # Assign variable 
    df = pd.read_csv('https://drive.google.com/uc?export=download&id=1T5zczOftHU3BXFXFb8czAvpN8uWmBp0C')
    # return it
    return df

# =======================================================================================================
# Acquire and Prepare END
# Exploration START
# =======================================================================================================

def dist_over_time(avg_by_yr):
    '''
    dist_over_time takes in the pga tour pandas df data then plots
    the average driving distance over time and displays the percentage increase
    '''
    # define figure size
    plt.figure(figsize=(10, 6))
    
    # Plot the driving distance over time
    plt.plot(avg_by_yr.index, avg_by_yr['drive_avg']) 
    
    # Calculate percentage increase
    start_distance = avg_by_yr['drive_avg'].iloc[0]
    end_distance = avg_by_yr['drive_avg'].iloc[-1]
    percentage_increase = ((end_distance - start_distance) / start_distance) * 100
    
    # Add text for percentage increase
    plt.text(2021, 290, f'+ {percentage_increase:.2f}%')

    # titles and labels
    plt.title('Driving Distance Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Driving Distance (yards)')
    plt.show()

    
def get_avg_by_year(df):
    '''
    get_avg_by_year takes in the pga tour pandas df data then groups data
    by year using mean and returns the avg_by_yr subset for exploration.
    '''
    cols = ['drive_avg', 'par_4_avg', 'par_5_avg']
    avg_by_yr = df.groupby('year')[cols].mean()
    return avg_by_yr
        
def scoring_over_time(df):
    '''
    scoring_over_time takes in the pga tour pandas df grouped by year then plots
    the average scores over time
    '''
    # define figure size
    plt.figure(figsize=(10, 6))

    # Plot scoring over time
    sns.lineplot(x=df.index, y='par_4_avg', data=df, label='Par 4 Avg')
    sns.lineplot(x=df.index, y='par_5_avg', data=df, label='Par 5 Avg')

    # Calculate trend lines and percentage change
    z_par_4 = np.polyfit(df.index, df['par_4_avg'], 1)
    p_par_4 = np.poly1d(z_par_4)
    trend_par_4 = p_par_4(df.index)
    change_par_4 = ((df['par_4_avg'].iloc[-1] - df['par_4_avg'].iloc[0]) / df['par_4_avg'].iloc[0]) * 100

    z_par_5 = np.polyfit(df.index, df['par_5_avg'], 1)
    p_par_5 = np.poly1d(z_par_5)
    trend_par_5 = p_par_5(df.index)
    change_par_5 = ((df['par_5_avg'].iloc[-1] - df['par_5_avg'].iloc[0]) / df['par_5_avg'].iloc[0]) * 100

    # Plot trend lines
    sns.lineplot(x=df.index, y=trend_par_4, label='Par 4 Trend', color='green', alpha=.5)
    sns.lineplot(x=df.index, y=trend_par_5, label='Par 5 Trend', color='green', alpha=.5)

    # Add text for percentage change
    plt.text(df.index[0], df['par_4_avg'].iloc[0], f'{change_par_4:.2f}%')
    plt.text(df.index[0], df['par_5_avg'].iloc[0], f'{change_par_5:.2f}%')

    # titles and labels
    plt.title('Par 4 and 5 Scoring Over Time')
    plt.xlabel('Year')
    plt.ylabel('Scoring')
    plt.legend()
    plt.show()
    
def get_ball_changes(avg_by_year):
    '''
    get_ball_changes takes in the pga tour pandas df grouped by year then plots
    the average distance over time with shaded regions displaying golfball changes
    '''
    # define fig size
    plt.figure(figsize=(16, 9))
    
    # plot drive distance avg
    plt.plot(avg_by_year.index, avg_by_year['drive_avg'])
    plt.xlabel('Year')
    plt.ylabel('Avg Drive Distance')
    plt.title('Driving Distance Over Time')

    # Shaded regions for golf ball changes
    plt.fill_between([1987, 1991], 250, 320, color='lightgray', alpha=0.3)
    plt.fill_between([1992, 1994], 250, 320, color='lightblue', alpha=0.3)
    plt.fill_between([1995, 2000], 250, 320, color='lightgreen', alpha=0.3)
    plt.fill_between([2001, 2006], 250, 320, color='lightyellow', alpha=0.3)
    plt.fill_between([2007, 2023], 250, 320, color='lightpink', alpha=0.3)

    # Text annotations for golf ball changes
    plt.text(1989, 270, 'Balata Cover', fontsize=10, ha='center')
    plt.text(1993, 280, 'Urethane Cover', fontsize=10, ha='center')
    plt.text(1997, 290, 'Multi-Layer Construction', fontsize=10, ha='center')
    plt.text(2004, 300, 'Low Compression Balls', fontsize=10, ha='center')
    plt.text(2014, 310, 'Improved Aerodynamics', fontsize=10, ha='center')

    plt.show()

    plt.show()    
    
def get_club_changes(avg_by_year):
    '''
    get_club_changes takes in the pga tour pandas df grouped by year then plots
    the average distance over time with shaded regions displaying golf club changes
    '''
    # define fig size
    plt.figure(figsize=(16, 9))
    
    # plot drive distance avg
    plt.plot(avg_by_year.index, avg_by_year['drive_avg'])
    plt.xlabel('Year')
    plt.ylabel('Avg Drive Distance')
    plt.title('Equipment improvements vs. Driving Distance')

    # Shaded regions for equipment changes
    plt.fill_between([1987, 1992], 250, 320, color='lightgray', alpha=0.3)
    plt.fill_between([1992, 1999], 250, 320, color='lightblue', alpha=0.3)
    plt.fill_between([1999, 2004], 250, 320, color='lightgreen', alpha=0.3)
    plt.fill_between([2004, 2010], 250, 320, color='lightyellow', alpha=0.3)
    plt.fill_between([2010, 2023], 250, 320, color='lightpink', alpha=0.3)

    # Text annotations for equipment changes
    plt.text(1990, 270, 'Metal Woods   ', fontsize=10, ha='center')
    plt.text(1995, 280, 'Graphite Shafts', fontsize=10, ha='center')
    plt.text(2001, 290, '     Titanium Drivers', fontsize=10, ha='center')
    plt.text(2007, 300, 'Hybrid Clubs', fontsize=10, ha='center')
    plt.text(2016, 310, 'Adjustable Clubs', fontsize=10, ha='center')

    plt.show()


#######################################

def get_par4_reg_analysis(df):
    '''
    get_par4_reg_analysis takes in the pga tour pandas df, splits the data into train
    and test, then fits a linear regression model to calculate coefficient, mse, and determination
    for par 4 data
    '''
    # Define the predictor variable and the target variable
    X = df[['drive_avg']]
    y = df['par_4_avg']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a linear regression object
    regressor = LinearRegression()

    # Train the model using the training sets
    regressor.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regressor.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regressor.coef_)
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))



def get_par5_reg_analysis(df):   
    '''
    get_par5_reg_analysis takes in the pga tour pandas df, splits the data into train
    and test, then fits a linear regression model to calculate coefficient, mse, and determination
    for par 5 data
    '''
    # Define the predictor variable and the target variable
    X = df[['drive_avg']]
    y = df['par_5_avg']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a linear regression object
    regressor = LinearRegression()

    # Train the model using the training sets
    regressor.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regressor.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regressor.coef_)
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

