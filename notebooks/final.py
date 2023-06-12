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

from datetime import datetime

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
    plt.title('Golf Ball Improvements vs. Driving Distance')

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
    plt.title('Equipment Improvements vs. Driving Distance')

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

#######-------------####### ------- POST MVP ------- ########----------##########

def load_data(file1, file2):
    
    ''' This function loads in two csv files and returns
    them a two separate dataframes'''
    
    # Load the files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    return df1, df2

def merge_data(df1, df2, key1, key2):
    
    """ This functions takes in two dataframes and two columns.
    and formats the text in both columns of the second dataframe. 
    
    The two dataframes are then merged on the two new formatted columns.
    One dataframes column had names first then last and the other dataframe had
    names last then first. 
    
    The lambda function corrected this to allow the merge.
    A merged data set is returned"""
    
    # Reformat the player names in the second dataframe
    df2[key2] = df2[key2].apply(lambda x: ' '.join(x.split(', ')[::-1]))

    # Merge the datasets
    merged_data = pd.merge(df1, df2, left_on=key1, right_on=key2, how='inner')

    return merged_data

def calculate_age_and_clean(df, dob_column, year_column, drop_column):
    
    """ Prepares DOB columns for timeseries, calculates age of players and drops nulls"""

    # Convert the 'DOB' column to datetime format
    df[dob_column] = pd.to_datetime(df[dob_column], format='%m/%d/%y')
    
    # Adjust the 'DOB' column so that the years are correctly interpreted as being in the 1900s
    df[dob_column] = df[dob_column].apply(lambda x: x if x.year < 2023 else datetime(x.year - 100, x.month, x.day))

    # Calculate the players' ages for each year of data
    df['age'] = df[year_column] - df[dob_column].dt.year
    
     # Removed duplicate player column
    df = df.drop(columns = drop_column)

    # Handle missing values
    df = df.dropna()

    return df

def predict_age_to_reach_drive_avg_positive_coef(merged_data):
    """
    This function predicts the age at which each player is expected to reach a drive_avg of 317 using a linear regression model.
    Only players with a positive coefficient are considered.

    Parameters:
    merged_data (DataFrame): The merged data of 'pga_scoring_and_drive.csv' and 'player_bio.csv'.

    Returns:
    dict: A dictionary with player names as keys and the predicted age to reach drive_avg of 317 as values.
    """
    # Initialize a dictionary to store the results
    results = {}

    # Loop over each player
    for player in merged_data['player'].unique():
        # Filter the data for the current player
        player_data = merged_data[merged_data['player'] == player]

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(player_data[['age']], player_data['drive_avg'])

        # Only consider players with a positive coefficient
        if model.coef_ > 0:
            # Predict the age at which the player is expected to reach a drive_avg of 317
            age_317 = (317 - model.intercept_) / model.coef_

            # Store the result if the player is expected to reach a drive_avg of 317 before turning 50
            if age_317 < 50:
                results[player] = age_317

    return results


def results_to_csv(results, output_filename):
    
    """ Function creates a dataframe from player and age dictionary, then 
    creates and adds column names before saving to a csv file"""
    
    # Convert the results dictionary to a DataFrame
    df_results = pd.DataFrame(results)

    # Melt the DataFrame to a long format
    df_results = df_results.melt()

    # Rename the columns
    df_results = df_results.rename(columns={"variable": "player", "value": "age"})

    # Save the DataFrame to a CSV file
    df_results.to_csv(output_filename)

    return df_results

############################ Merged DataFrame ########################################

def merge_dataframes(file1, file2, common_column):
    """
    This function loads two csv files and merges them on a common column.

    Parameters:
    file1 (str): The name of the first csv file.
    file2 (str): The name of the second csv file.
    common_column (str): The name of the common column to merge on.

    Returns:
    DataFrame: The merged dataframe.
    """
    # Load the first csv file
    df1 = pd.read_csv(file1)

    # Load the second csv file
    df2 = pd.read_csv(file2)

    # Rename the 'name' column in df2 to 'player'
    df2.rename(columns={'name': 'player'}, inplace=True)

    # Merge the two dataframes
    merged_df = pd.merge(df1, df2, on=common_column, how='inner')

    return merged_df

def process_and_save_dataframe(df, output_filename):
    """
    This function drops duplicate and unneeded columns from a dataframe, creates a new column 'predicted_years'
    that holds the difference between 'age_y' and 'age_x', and saves the dataframe to a csv file.

    Parameters:
    df (DataFrame): The dataframe to process.
    output_filename (str): The name of the output csv file.
    """
    # Drop duplicate and unneeded columns
    df = df.drop(columns=['Unnamed: 0'])

    # Create a new column 'predicted_years' that holds the difference between 'age_y' and 'age_x'
    df['predicted_years'] = df['age_y'] - df['age_x']

    # Save the dataframe to a csv file
    df.to_csv(output_filename, index=False)
    
    return df


def plot_predicted_years(df):
    """
    This function plots a histogram of the 'predicted_years' column in the dataframe.

    Parameters:
    df (DataFrame): The dataframe to plot.
    """
    # Plot a histogram of predicted_years
    plt.figure(figsize=(10, 6))
    sns.histplot(df['predicted_years'], kde=True, bins=30)
    plt.xlabel('Predicted Years')
    plt.ylabel('Players')
    plt.title('Distribution of Predicted Years to Reach Drive Average of 317')
    plt.show()

def plot_predicted_age_groups(df):
    """
    This function creates bins for 'age_y' and plots a histogram of the 'age_y' column in the dataframe.

    Parameters:
    df (DataFrame): The dataframe to plot.
    """
    # Create bins for 'age_y'
    bins = [20, 30, 40, 50]
    labels = ['20-29', '30-39', '40-49']
    df['age_y_bin'] = pd.cut(df['age_y'], bins=bins, labels=labels)

    # Plot a histogram of 'age_y' with bins
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age_y_bin'], bins=30)
    plt.xlabel('Predicted Age Groups')
    plt.ylabel('Players')
    plt.title('Distribution of Predicted Age to Reach Drive Average of 317')
    plt.show()

def plot_predicted_years_bins(df):
    """
    This function creates bins for 'predicted_years' and plots a histogram of the 'predicted_years' column in the dataframe.

    Parameters:
    df (DataFrame): The dataframe to plot.
    """
    # Create bins for 'predicted_years'
    bins = [0, 10, 20, 30]
    labels = ['0-10', '10-20', '20-30']
    df['predicted_years_bin'] = pd.cut(df['predicted_years'], bins=bins, labels=labels)

    # Plot a histogram of 'predicted_years' with bins
    plt.figure(figsize=(10, 6))
    sns.histplot(df['predicted_years_bin'], bins=30)
    plt.xlabel('Predicted Years')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Years to Reach Drive Average of 317')
    plt.show()

def plot_drive_avg_percentiles(df):
    """
    This function calculates the percentiles for 'drive_avg' and plots a bar chart of the percentiles.

    Parameters:
    df (DataFrame): The dataframe to plot.
    """
    # Calculate the percentiles for 'drive_avg'
    top_10 = df['drive_avg'].quantile(0.9)
    bottom_10 = df['drive_avg'].quantile(0.1)
    middle_80 = df['drive_avg'].quantile(0.5)

    # Create a dataframe for the percentiles
    percentiles_df = pd.DataFrame({'Percentile': ['Top 10%', 'Middle 80%', 'Bottom 10%'],
                                   'Drive Average': [top_10, middle_80, bottom_10]})

    # Plot a bar chart of the percentiles
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Percentile', y='Drive Average', data=percentiles_df, palette='Blues')
    plt.title('Drive Average for Top 10%, Middle 80%, and Bottom 10%')
    plt.show()

def plot_drive_avg_and_par_5_avg_percentiles(df):
    """
    This function calculates the percentiles for 'drive_avg' and 'par_5_avg' and plots a bar chart of the percentiles.

    Parameters:
    df (DataFrame): The dataframe to plot.
    """
    # Calculate the percentiles for 'drive_avg' and 'par_5_avg'
    top_10_drive = df['drive_avg'].quantile(0.9) / 100
    middle_80_drive = df['drive_avg'].quantile(0.5) / 100
    bottom_10_drive = df['drive_avg'].quantile(0.1) / 100

    top_10_par_5 = df['par_5_avg'].quantile(0.9)
    middle_80_par_5 = df['par_5_avg'].quantile(0.5)
    bottom_10_par_5 = df['par_5_avg'].quantile(0.1)

    # Create a dataframe for the percentiles
    percentiles_df = pd.DataFrame({'Percentile': ['Top 10%', 'Middle 80%', 'Bottom 10%'],
                                   'Drive Average': [top_10_drive, middle_80_drive, bottom_10_drive],
                                   'Par 5 Average': [top_10_par_5, middle_80_par_5, bottom_10_par_5]})

    # Plot a bar chart of the percentiles
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Percentile', y='Drive Average', data=percentiles_df, palette='Blues')
    sns.barplot(x='Percentile', y='Par 5 Average', data=percentiles_df, palette='Blues')
    plt.title('Drive Average and Par 5 Average for Top 10%, Middle 80%, and Bottom 10%')
    plt.legend(['Drive Average (scaled)', 'Par 5 Average'])
    plt.show()

def create_percentile_dataframes(merged_df, top_10, bottom_10):
    top_10_df = merged_df[merged_df['drive_avg'] >= top_10]
    middle_80_df = merged_df[(merged_df['drive_avg'] < top_10) & (merged_df['drive_avg'] > bottom_10)]
    bottom_10_df = merged_df[merged_df['drive_avg'] <= bottom_10]
    return top_10_df, middle_80_df, bottom_10_df


def calculate_and_plot_averages(top_10_df, middle_80_df, bottom_10_df, top_10, middle_80, bottom_10):
    # Calculate the average 'par_5_avg' for each percentile
    top_10_par_5_avg = top_10_df['par_5_avg'].mean()
    middle_80_par_5_avg = middle_80_df['par_5_avg'].mean()
    bottom_10_par_5_avg = bottom_10_df['par_5_avg'].mean()

    # Create a dataframe for the averages
    averages_df = pd.DataFrame({'Percentile': ['Top 10%', 'Middle 80%', 'Bottom 10%'],
                                'Drive Average': [top_10, middle_80, bottom_10],
                                'Par 5 Average': [top_10_par_5_avg, middle_80_par_5_avg, bottom_10_par_5_avg]})

    # Round off the 'par_5_avg' in the dataframe
    averages_df['Par 5 Average'] = averages_df['Par 5 Average'].round(2)

    # Plot a bar chart of 'par_5_avg' vs 'drive_avg'
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Par 5 Average', y='Drive Average', hue='Percentile', data=averages_df, palette='Blues')
    plt.title('Drive Average vs Par 5 Average for Top 10%, Middle 80%, and Bottom 10%')
    plt.show()



def calculate_and_plot_averages(merged_df):
    # Calculate the percentiles for 'drive_avg'
    top_10 = merged_df['drive_avg'].quantile(0.9)
    bottom_10 = merged_df['drive_avg'].quantile(0.1)
    middle_80 = merged_df['drive_avg'].quantile(0.5)

    # Create a new dataframe for the percentiles
    top_10_df = merged_df[merged_df['drive_avg'] >= top_10]
    middle_80_df = merged_df[(merged_df['drive_avg'] < top_10) & (merged_df['drive_avg'] > bottom_10)]
    bottom_10_df = merged_df[merged_df['drive_avg'] <= bottom_10]

    # Calculate the average 'par_5_avg' for each percentile
    top_10_par_5_avg = top_10_df['par_5_avg'].mean()
    middle_80_par_5_avg = middle_80_df['par_5_avg'].mean()
    bottom_10_par_5_avg = bottom_10_df['par_5_avg'].mean()

    # Create a dataframe for the averages
    averages_df = pd.DataFrame({'Percentile': ['Top 10%', 'Middle 80%', 'Bottom 10%'],
                                'Drive Average': [top_10, middle_80, bottom_10],
                                'Par 5 Average': [top_10_par_5_avg, middle_80_par_5_avg, bottom_10_par_5_avg]})

    # Round off the 'par_5_avg' in the dataframe
    averages_df['Par 5 Average'] = averages_df['Par 5 Average'].round(2)

    # Plot a bar chart of 'par_5_avg' vs 'drive_avg'
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Par 5 Average', y='Drive Average', hue='Percentile', data=averages_df, palette='Blues')
    plt.title('Drive Average vs Par 5 Average for Top 10%, Middle 80%, and Bottom 10%')
    plt.show()

def calculate_and_plot_averages_par_4(merged_df):
    # Calculate the percentiles for 'drive_avg'
    top_10 = merged_df['drive_avg'].quantile(0.9)
    bottom_10 = merged_df['drive_avg'].quantile(0.1)
    middle_80 = merged_df['drive_avg'].quantile(0.5)

    # Create a new dataframe for the percentiles
    top_10_df = merged_df[merged_df['drive_avg'] >= top_10]
    middle_80_df = merged_df[(merged_df['drive_avg'] < top_10) & (merged_df['drive_avg'] > bottom_10)]
    bottom_10_df = merged_df[merged_df['drive_avg'] <= bottom_10]

    # Calculate the average 'par_5_avg' for each percentile
    top_10_par_5_avg = top_10_df['par_4_avg'].mean()
    middle_80_par_5_avg = middle_80_df['par_4_avg'].mean()
    bottom_10_par_5_avg = bottom_10_df['par_4_avg'].mean()

    # Create a dataframe for the averages
    averages_df = pd.DataFrame({'Percentile': ['Top 10%', 'Middle 80%', 'Bottom 10%'],
                                'Drive Average': [top_10, middle_80, bottom_10],
                                'Par 4 Average': [top_10_par_5_avg, middle_80_par_5_avg, bottom_10_par_5_avg]})

    # Round off the 'par_5_avg' in the dataframe
    averages_df['Par 4 Average'] = averages_df['Par 4 Average'].round(2)

    # Plot a bar chart of 'par_5_avg' vs 'drive_avg'
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Par 4 Average', y='Drive Average', hue='Percentile', data=averages_df, palette='Blues')
    plt.title('Drive Average vs Par 4 Average for Top 10%, Middle 80%, and Bottom 10%')
    plt.show()

def load_and_prep_pmvp(file1, file2):
    '''
    load_and_prep_pmvp loads 2 csv's, preps, merges, and
    returns the merged df
    '''
    # Load the data
    pga_scoring_and_drive, player_bio = load_data(file1, file2)
    # Format the player names in the second dataframe and merge the datasets
    merged_data = merge_data(pga_scoring_and_drive, player_bio, 'player', 'Players')
    # Corrected DOB, calculated players age, and removed duplicates
    merged_data = calculate_age_and_clean(merged_data, 'DOB', 'year','Players')
    # return df
    return merged_data


def load_pmvp(file):
    '''
    reads in csv as a pandas df
    '''
    merged_data = pd.read_csv(file)
    return merged_data







