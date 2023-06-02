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
from sklearn.tree import DecisionTreeRegressor
from math import sqrt 

import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing

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

    
def get_df_by_year(df):
    '''
    df_by_year takes in the pga tour pandas df data then groups data
    by year using mean and returns the avg_by_yr subset for exploration.
    '''
    cols = ['drive_avg', 'par_4_avg', 'par_5_avg']
    df_by_year = df.groupby('year')[cols].mean()
    df_by_year = pd.DataFrame(df_by_year)
    return df_by_year
        
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

#########################

def tts(df):
    '''
    tts takes in a pandas df and splits the data into train, validate, 
    and test at a 70/20/10 split stratifying on the index.
    '''
    # define split sizes
    train_size = int(len(df) * .7)
    validate_size = int(len(df) * .2)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    validate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]

    return train, validate, test

def evaluate(target_var, validate, yhat_df):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 2 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 2)
    return rmse

def plot_and_eval(target_var, train, validate, yhat_df):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var, validate, yhat_df)
    print(target_var, '-- RMSE: {:.2f}'.format(rmse))
    plt.show()
    
    
    
# function to store rmse for comparison purposes
def append_eval_df(model_type, target_var, validate, yhat_df, eval_df):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var, validate, yhat_df)
    d = {'model_type': [model_type], 'target_var': [target_var], 'rmse': [rmse]}
    d = pd.DataFrame(d)
    # call function to creat empty df
    
    return pd.concat([eval_df, d])    
    

        
def compute_moving_avg(train, validate):
    '''
    compute_moving_avg takes in train, validate,
    computes the moving avg for 5 periods, and appends to the eval df    
    '''
    # empty df for results
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    
    # define periods
    periods = [3, 6, 9]

    for p in periods: 
        rolling_drive = round(train['drive_avg'].rolling(p).mean()[-1], 2)
        rolling_par4 = round(train['par_4_avg'].rolling(p).mean()[-1], 2)
        rolling_par5 = round(train['par_5_avg'].rolling(p).mean()[-1], 2)

        yhat_df = pd.DataFrame({'drive_avg': rolling_drive,
                                'par_4_avg': rolling_par4,
                                'par_5_avg': rolling_par5},
                                 index=validate.index)

        model_type = str(p) + '_year_moving_avg'
        for col in train.columns:
            eval_df = append_eval_df(model_type = model_type,
                                    target_var = col, validate = validate,
                                     yhat_df = yhat_df, eval_df = eval_df)
    return eval_df

def model_prep():
    '''
    re-acquires data, converts to date time, drops unnecessary cols, splits data
    '''
    df = acquire_data()
    # Reassign the year column to be a datetime type
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    # Sort rows by the date and then set the index as that date
    df = df.set_index("year").sort_index()
    # identify cols we're moving into modeling
    cols = ['drive_avg', 'par_4_avg', 'par_5_avg']
    # group by year
    df_by_year = df.groupby('year')[cols].mean()
    # convert to df
    df_by_year = pd.DataFrame(df_by_year)
    # split data
    train, validate, test = tts(df_by_year)
    # return
    return train, validate, test


def train_val_best_model(train, validate):
    '''
    
    '''
    # create empty df for predicition
    yhat_df = pd.DataFrame({'drive_avg': 0,
                        'par_4_avg': 0,
                        'par_5_avg': 0},
                         index=validate.index)
    # train model and test on validate set
    for col in train.columns:
        model = Holt(train[col], exponential=True, damped=False)
        model = model.fit(optimized=True, smoothing_slope=.5, smoothing_level = .7)
        yhat_values = model.predict(start = validate.index[0],
                                  end = validate.index[-1])
        yhat_df[col] = round(yhat_values, 2)
    # plot and evaluate
    for col in train.columns:
        return plot_and_eval(target_var = col, train = train,
                            validate = validate, yhat_df = yhat_df)


    
def final_plot(target_var, train, validate, test, yhat_df):
    plt.figure(figsize=(12,4))
    plt.plot(train[target_var], color='#377eb8', label='train')
    plt.plot(validate[target_var], color='#ff7f00', label='validate')
    plt.plot(test[target_var], color='#4daf4a',label='test')
    plt.plot(yhat_df[target_var], color='#a65628', label='yhat')
    plt.legend()
    plt.title(target_var)
    plt.show()    
    
    

def test_best_model(train, validate, test):
    '''
    
    '''
    # create empty df for predicition
    yhat_df = pd.DataFrame({'drive_avg': 0,
                        'par_4_avg': 0,
                        'par_5_avg': 0},
                         index=test.index)
    # train model and test on validate set
    for col in train.columns:
        model = Holt(train[col], exponential=True, damped=False)
        model = model.fit(optimized=True, smoothing_slope=.5, smoothing_level = .7)
        yhat_values = model.predict(start = test.index[0],
                                  end = test.index[-1])
        yhat_df[col] = round(yhat_values, 2)
    # plot and evaluate
    rmse_drive_avg = sqrt(mean_squared_error(test['drive_avg'], 
                                           yhat_df['drive_avg']))

    rmse_par_4_avg = sqrt(mean_squared_error(test['par_4_avg'], 
                                           yhat_df['par_4_avg']))

    rmse_par_5_avg = sqrt(mean_squared_error(test['par_5_avg'], 
                                           yhat_df['par_5_avg']))

    print('FINAL PERFORMANCE OF MODEL ON TEST DATA')
    print('rmse- drive_avg: ', rmse_drive_avg)
    print('rmse- par_4_avg: ', rmse_par_4_avg)
    print('rmse- par_5_avg: ', rmse_par_5_avg)
    for col in train.columns:
        final_plot(col, train, validate, test, yhat_df)






