#main module: it's the entry point to call all the logical blocks of our application
from matplotlib import pyplot as plt

from eda import get_data, print_data_general_info, plot_time_series, plot_distributions
from arima import arima

if __name__ == '__main__':
    #1. Exploratory data analysis
    #   1.1. get the dataframe (open file, convert columns...)
    df = get_data()
    #   1.2. print data general information (e.g. length, frequencies, null values, basic stats)
    print_data_general_info(df)

    #2. EDA (Exploratory Data Analysis)
    #   2.1. plot time series
    plot_time_series(df)
    #   2.2. plot distributions
    plot_distributions(df)