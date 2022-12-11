import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load two csv files given by the user and merge to a single dataframe

    Input:
    messages_filepath       filepath to message csv file
    categories_filepath     filepath to categories csv file

    Returns:
    df      dataframe merging categories and messages

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    return pd.merge(messages,categories)

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    '''
    clean_data
    Clean the dataset.

    The column with message categories is separated into multiple columns with value 0/1.
    Some strange values in the 'related' column are fixed.
    Drop duplicated rows
    
    Input:
    df       dataframe to clean

    Returns:
    df      dataframe cleaned
    '''
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

    # convert column from string to numeric
    categories = categories.astype('int')

    # Replacig value =2 with value=0 in related column
    categories['related'] = categories['related'].map({1:1,0:0, 2:0})

    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    # Drop the original column of the dataframe
    df = df.drop('original',axis=1)

    return df


def save_data(df, database_filename):
    '''
    save_data
    Save data into a database

    Input:
    df      dataframe to be stored in the database
    database_filename      database name to store the dataframe

    Returns: None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Disaster_Resp', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()