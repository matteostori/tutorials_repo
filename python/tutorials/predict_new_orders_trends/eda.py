# module including functions for exploratory data analysis
from os import listdir
from os.path import join
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# entry point function to go through all the steps in this module:
# 1.1. clean up data
# 1.2. analyze input distribution
# 1.3. print output file
def start_eda():
    df_orders = clean_up_data()
    analyze_input_distribution(df_orders)
    #print_output_data(df_orders)

#   1.1 clean up input data
def clean_up_data():
    #------> open all input files and create a single dataframe
    print('Opening all input csv files')
    inputDir = '../tutorials_repo/data_sources/orders_products_dataset' #our input directory
    df_orders = pd.DataFrame() #let's create an empty dataframe to put together all single file
    for f in listdir(inputDir):
        if f.endswith('.csv'): #we need to pay attention and process only csv files
            inputFile = join(inputDir, f) #we need to create a string containing the input dir path + the file name of the current iteration
            df = pd.read_csv(inputFile) #open the current file
            df_orders = pd.concat([df_orders,  df], ignore_index=True) #we "append" the new file in the df_order variable
            print('Final dataframe contains now {} rows, current dataframe from file {} added ({} rows)'.format(len(df_orders), f, len(df)))

    #------> delete unused columns and empty rows
    df_orders = df_orders.drop(columns=['Purchase Address'])
    print('Null values...')
    print(df_orders.isna().sum()) #print number of rows, for each colums, containing null values
    df_orders = df_orders.dropna() #remove rows containing null values (# rows = print in previous line)
    # as there are rows containing the headers (i.e. looks like several different files have been appended to each other)
    # we need to remove those rows, which actually don't contain valid data
    # we use the loc function to select all rows in which the column named 'Quantity Ordered' doesn't contain
    # the string 'Quantity Ordered'; this way the new df_orders contains only valid values
    df_orders = df_orders.loc[df_orders['Quantity Ordered'] != 'Quantity Ordered']
    # now we need to tell explicitely tell the dataframe that some columns contains 'numbers', since at the beginning
    # those rows contained both numbers and strings, the dataframe labeled them generically as 'object'
    # since the column 'Quantity Ordered' contains integers, we convert to to 'int' while 'Price Each' to float (ie. with decimal values)
    df_orders['Quantity Ordered'] = df_orders['Quantity Ordered'].astype('int')
    df_orders['Price Each'] = df_orders['Price Each'].astype('float')
    df_orders['Order Date'] = pd.to_datetime(df_orders['Order Date'], format='%m/%d/%y %H:%M')
    df_orders['Order year'] = pd.DatetimeIndex(df_orders['Order Date']).year
    df_orders['Order month'] = pd.DatetimeIndex(df_orders['Order Date']).month
    
    print('Final dataframe column names: {}; number of rows: {}'.format(df_orders.columns, len(df_orders)))

    return df_orders

#   1.2. analyze input distribution
def analyze_input_distribution(df_orders):
    print('Printing general stats')
    
    # print how many rows we have with distinct products
    print('Print how many orders we have for each distinct product')
    df_orders_products = df_orders['Product'].value_counts()
    print(df_orders_products)

    # print the sum, mean, min, max, variance of quantities ordered and unit price
    print('Print basics stats for quantity ordered and price each columns')
    print(df_orders.groupby('Product').agg(
        {'Quantity Ordered':['sum', 'min', 'max', 'var'], 'Price Each': ['mean', 'min', 'max', 'var']}
    ))

    corr = df_orders.corr()
    print(corr)
    plt.figure(1)
    sns.heatmap(corr, 
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values, linewidths=1.5, annot=True, fmt='.2f')
    plt.show()

#   1.3. write output file
def print_output_data(df_orders):
    df_orders.to_excel('../tutorials_repo/python/tutorials/predict_new_orders_trends/output/Orders_2019.xlsx')