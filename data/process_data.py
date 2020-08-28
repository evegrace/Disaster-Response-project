import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on='id', how='left')
    return df


def clean_data(df):

    # Expand categories into separate columns
    categories = df.categories.str.split(';', expand=True)
    colnames = categories.iloc[0].str.split('-', expand=True)[0].tolist()
    categories.columns = colnames
    
    # Clean values and convert to numeric if the category is not constant
    for column in categories.columns:
        if categories[column].nunique() > 1:
            categories[column] = categories[column].apply(lambda r: r[-1]).astype(int)
        else:
            categories.drop(column, axis=1, inplace=True)
            
                # Combine original df and expanded categories
    return pd.concat([df.drop('categories', axis=1), categories], axis=1).drop_duplicates()
    


def save_data(df, database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('dmessages', engine, index=False, if_exists='replace')
    return engine


def main():

    '''
    This file is the ETL pipeline that cleans the data and stores it in a SQLite database.
    From this project's root directory, run this file with:
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/messages.db
    '''

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:

        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'messages.db')


if __name__ == '__main__':
    main()