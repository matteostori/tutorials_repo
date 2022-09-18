# module including functions for exploratory data analysis
from os import listdir
from os.path import join
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing._label import LabelEncoder

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
    
    #cities are in the form "917 1st St, Dallas, TX 75001"
    df_orders['City'] = df_orders['Purchase Address'].str.split(',').str[1]
    df_orders['State ZIP'] = df_orders['Purchase Address'].str.split(',').str[2]
    df_orders['State'] = df_orders['State ZIP'].str.split(' ').str[1]
    df_orders = df_orders.drop(columns=['State ZIP'])

    print('Final dataframe column names: {}; number of rows: {}'.format(df_orders.columns, len(df_orders)))

    return df_orders

#   1.2. analyze input distribution
def analyze_input_distribution(df_orders):
    print('Printing general stats')
    
    #------> print how many rows we have with distinct products
    print('Print how many orders we have for each distinct product')
    print(df_orders['Product'].value_counts())

    #------> print distinct values by column
    print('Print how many distinct values we have for each column')
    print(df_orders.value_counts())

    #------> print the sum, mean, min, max, variance of quantities ordered and unit price
    print('Print basics stats for quantity ordered and price each columns')
    print(df_orders.groupby('Product').agg(
        {'Quantity Ordered':['sum', 'min', 'max', 'var'], 'Price Each': ['mean', 'min', 'max', 'var']}
    ))

    #------> plot the total amount of items ordered for each month of the year, by product

    # calculate the sum of the ordered quantities for each product, by month
    # the resulting dataframe will contain thre columns (Product, Order month and Quantity Ordered)
    df_orders_products = df_orders.groupby(['Product', 'Order month' ], as_index=False)[['Quantity Ordered']].sum()

    # variables to control chart creation
    plot_chart = 1 # number of chart
    plt.figure(plot_chart)
    plot_chart = plot_chart + 1
    i = 1 # used to check how many products we have analyzed in the for loop            
    products_per_chart = 5 #as we have many products, we put only 5 lines per chart
    for product in df_orders_products['Product'].unique():
        if i % products_per_chart == 0:
            plt.legend()
            plt.ylabel('Quantity ordered (# items)')
            plt.xlabel('Months')
            plt.title('Sum of ordered quantities by product - 2019')
            plt.figure(plot_chart)
            plot_chart = plot_chart + 1

        print('Preparing chart for product {}'.format(product))
        df_product = df_orders_products.loc[df_orders_products['Product'] == product]
        plt.plot(df_product['Order month'], df_product['Quantity Ordered'], 'o-', label=product)
        
        i = i + 1

    plt.legend()
    plt.ylabel('Quantity ordered (# items)')
    plt.xlabel('Months')
    plt.title('Sum of ordered quantities by product - 2019')

    label_encoder = LabelEncoder()
    df_orders['State_label'] = label_encoder.fit_transform(df_orders['State'])
    df_orders['City_label'] = label_encoder.fit_transform(df_orders['City'])

    df_orders_products = df_orders.groupby(['Product', 'Order month', 'State', 'City'], 
                                                    as_index=False)[['Quantity Ordered']].sum()
                        

    g = sns.relplot(x="Order month", y="Quantity Ordered", kind="line", hue="State", data=df_orders_products)
    #g.figure.autofmt_xdate()

   # sns.catplot(x="City", y="Quantity Ordered", data=df_orders_products)
   # sns.relplot(x="Order month", y="Quantity Ordered", data=df_orders_products, hue="Product")
   # sns.catplot(x="State", y="Quantity Ordered", kind="boxen", data=df_orders_products)
   
    #print('Seaborn plot')
    #fig, ax = plt.subplots()
    #f, ax = plt.subplots(figsize=(7, 5))
    #sns.displot(df_orders, x="Quantity Ordered", bins=50, hue="Product")
    #print('Seaborn plot end')

    #fig, ax = plt.subplots(figsize=(8, 8))
    #ax = sns.boxplot(x='State', y='Quantity Ordered', data=df_orders, ax=ax); ax.set_ylabel("Quantity Ordered [# pieces]");
    #ax.set_title("Quantities ordered by state");

   # corr = df_orders_products.corr()
   # print('Plot correlation matrix')
   # plot_chart = plot_chart + 1
   # plt.figure(plot_chart)
   # sns.heatmap(corr, 
   #     xticklabels=corr.columns.values,
   #     yticklabels=corr.columns.values, linewidths=1.5, annot=True, fmt='.2f')
    
    plt.draw()


#   1.3. write output file
def print_output_data(df_orders):
    df_orders.to_excel('../tutorials_repo/python/tutorials/predict_new_orders_trends/output/Orders_2019.xlsx')