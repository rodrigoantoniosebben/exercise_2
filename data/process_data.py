import sys
import sqlite3
import pandas as pd
import sqlalchemy as sql


def load_data(messages_filepath, categories_filepath):
    ''''
        load messages and categories and merge them

        INPUT:
        messages_filepath: Path to the messages file
        categories_filepath: Path to the categories file

        OUTPUT:
        df: The dataframe with the merged information
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    ''''
        clean the dataframe

        INPUT: 
        df: the dataframe that will be cleaned

        OUTUPT:
        df: the dataframe cleaned
    '''
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.str[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames.iloc[0]

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df.drop(columns='categories', axis=0, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    # forcing all related data to be binary
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    return df


def save_data(df, database_filename):
    ''''
        save the dataframe to a sql file
        INPUT
            df: the dataframe
            database_filename: the file name of the sql file
    '''
    print(df.head())
    engine = sql.create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
