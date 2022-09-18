import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def get_data():
    #----------------> open input file
    input_file = '../tutorials_repo/data_sources/sales_dataset/sales_store_dataset.csv'
    df = pd.read_csv(input_file)
    #convert columns to datetime using day/month/year format
    df['Order Date'] = pd.to_datetime(df['Order Date'], format="%d/%m/%Y")
    df['Ship Date'] = pd.to_datetime(df['Order Date'], format="%d/%m/%Y")
    #create year and month columns for the order
    df['Order Year'] = pd.DatetimeIndex(df['Order Date']).year
    df['Order Month'] = pd.DatetimeIndex(df['Order Date']).month
    #create "yearmonth" column using the year and month columns and convert it to date type
    df["Order Date YearMonth"] = df["Order Year"].astype(str) + "-" + df["Order Month"].astype(str)
    df["Order Date YearMonth"] = pd.to_datetime(df["Order Date YearMonth"]).dt.date

    #sort the dataframe by order date and reset the index
    df = df.sort_values(by = ['Order Date'])
    df.reset_index(drop=True, inplace=True)

    return df

#print some data info
def print_data_general_info(df):
    #general info on the dataframe
    print(df.describe())
    #print number of rows, for each colums, containing null values
    print(f"Total null values: {df.isna().sum()}")
    #check the type of the order date column
    print(f"Order date type: {df['Order Date'].dtype}")
    #frequencies of columns
    print("-----------------------> 'Product name' column value counts:")
    print(df['Product Name'].value_counts())
    print("-----------------------> 'Region' column value counts:")
    print(df['Region'].value_counts())
    print("-----------------------> 'Segment' value counts:")
    print(df['Segment'].value_counts())
    print("-----------------------> 'Sub category' value counts:")
    print(df['Sub-Category'].value_counts())
    #first and last orders date
    print(f"First Order date: {df['Order Date'].iloc[0]}")
    print(f"Last Order date: {df['Order Date'].iloc[len(df)-1]}")

def plot_time_series(df):
    df1 = df.copy()
    df1.index = df1['Order Date']
    df2 = df1.groupby([pd.Grouper(freq='1M'), 'Region']).sum()
    df2 = df2.reset_index().pivot(index='Order Date', columns='Region', values='Sales')
    df2.plot()
    
    df3 = df1.groupby([pd.Grouper(freq='1M')])[['Sales']].sum()
    df3["rollmean_1"] = df3['Sales'].rolling(window=1).mean()
    df3["rollmean_3"] = df3['Sales'].rolling(window=3).mean()
    df3["rollmean_5"] = df3['Sales'].rolling(window=5).mean()
    df3.plot()

    g = sns.displot(df, x="Sales", kde=True, hue="Region")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title("Sales distribution by Region")
    g.figure.autofmt_xdate()
    sns.lineplot(x='Order Month', y='Sales', hue='Order Year', data=df)

    plt.show()

def plot_distributions(df):
    df = df.loc[df['Category'] == "Technology"]
    df_orders = df.groupby(['Order ID', 'Region', 'Order Year', 'Order Month'], as_index=False)[['Sales']].sum()
    plot_distribution_box_plot(df_orders, "Region", "Orders")

    for sc in df['Segment'].unique():
        print(f"----------------> Preparing chart for product {sc}")
        df_plot = df.loc[df['Segment'] == sc]
        print(df_plot.describe())
        #q = df_plot["Sales"].quantile(0.95)
        #df_plot = df_plot.loc[df_plot["Sales"] < q]
        print(f"{df_plot.describe()}, rows: {len(df_plot)}")
        plot_distribution_box_plot(df_plot, "Region", sc)

#plot distribution and box plot
def plot_distribution_box_plot(df_product, split, title):
    g = sns.displot(df_product, x="Sales", kde=True, hue=split)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title(title)
    g.figure.autofmt_xdate()
    plt.draw()
    
    g = sns.catplot(data=df_product, x=split, y="Sales", kind="box")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title(title)
    g.figure.autofmt_xdate()
    plt.draw()

    g = sns.relplot(x="Order Month", y="Sales", kind="line", hue="Order Year", data=df_product)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title(title)
    g.figure.autofmt_xdate()
    plt.show()