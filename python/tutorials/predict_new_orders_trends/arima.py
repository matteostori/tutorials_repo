from gettext import translation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing._label import LabelEncoder
import traceback
from pandas.tseries.frequencies import to_offset
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

def eda(df_product, split, title):
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

    # sns.lineplot(x='Order Month', y='Sales', hue='Order Year', data=df)

    g = sns.relplot(x="Order Month", y="Sales", kind="line", hue="Order Year", data=df_product)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title(title)
    g.figure.autofmt_xdate()
    plt.show()

if __name__ == '__main__':

    input_file = '../tutorials_repo/data_sources/sales_dataset/sales_store_dataset.csv'
    df = pd.read_csv(input_file) #open the current file
    df['Order Date'] = pd.to_datetime(df['Order Date'], format="%d/%m/%Y")
    df['Ship Date'] = pd.to_datetime(df['Order Date'], format="%d/%m/%Y")
    df['Order Year'] = pd.DatetimeIndex(df['Order Date']).year
    df['Order Month'] = pd.DatetimeIndex(df['Order Date']).month
    df["Order Date YearMonth"] = df["Order Year"].astype(str) + "-" + df["Order Month"].astype(str)
    df["Order Date YearMonth"] = pd.to_datetime(df["Order Date YearMonth"]).dt.date

    df = df.sort_values(by = ['Order Date'])
    df.reset_index(drop=True, inplace=True)
    print(df['Order Date'].dtype)
    print(df.columns)
    print(df.describe())
    print(df['Product Name'].value_counts())
    print(df['Region'].value_counts())
    print(df['Segment'].value_counts())
    print(df['Sub-Category'].value_counts())
    print(df['Order Date'].iloc[0])
    print(df['Order Date'].iloc[len(df)-1])

    print('Null values...')
    print(df.isna().sum()) #print number of rows, for each colums, containing null values

    # label_encoder = LabelEncoder()
    # df['Ship Mode int'] = label_encoder.fit_transform(df['Ship Mode'])
    # df['Customer ID int'] = label_encoder.fit_transform(df['Customer ID'])
    # df['Segment int'] = label_encoder.fit_transform(df['Segment'])
    # df['Country int'] = label_encoder.fit_transform(df['Country'])
    # df['City int'] = label_encoder.fit_transform(df['City'])
    # df['State int'] = label_encoder.fit_transform(df['State'])
    # df['Region int'] = label_encoder.fit_transform(df['Region'])
    # df['Product ID int'] = label_encoder.fit_transform(df['Product ID'])
    # df['Category int'] = label_encoder.fit_transform(df['Category'])
    # df['Sub-Category int'] = label_encoder.fit_transform(df['Sub-Category'])

    # corr = df.corr()
    # sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, annot_kws={'fontsize':11, 'fontweight':'bold'})
    # plt.show()

    #df_pivot = df.groupby(['Order Date', 'Region'], as_index='False')[['Sales']].sum()
    #print(df_pivot.describe())
    #g = sns.relplot(x="Order Date", y="Sales", kind="line", hue="Region", data=df_pivot)
    #g.figure.autofmt_xdate()
    # plt.figure(1)

    df1 = df.copy()
    df1.index = df1['Order Date']
    df2 = df1.groupby([pd.Grouper(freq='1M'), 'Region']).sum()
    df2 = df2.reset_index().pivot(index='Order Date', columns='Region', values='Sales')
    df2.plot()
    
    
    df3 = df1.groupby([pd.Grouper(freq='1M')])[['Sales']].sum()
    df4 = df3.copy()
    df4["rollmean_1"] = df4['Sales'].rolling(window=1).mean()
    df4["rollmean_3"] = df4['Sales'].rolling(window=3).mean()
    df4["rollmean_5"] = df4['Sales'].rolling(window=5).mean()
    df4.plot()

    #-----------------> Checking differencing order (i) of the Arima model / test for stationarity
    ADF_result = adfuller(df3['Sales'])
    print(f'ADF Statistic: {ADF_result[0]}')
    print(f'p-value: {ADF_result[1]}')
    
    df_diff = np.diff(df3['Sales'], n=1)
    ADF_result = adfuller(df_diff)
    print(f'ADF Statistic: {ADF_result[0]}')
    print(f'p-value: {ADF_result[1]}')
    
    plot_acf(df_diff, lags=30)
    plt.tight_layout()

    seasonal_decomp = seasonal_decompose(df3['Sales'], model="additive")
    seasonal_decomp.plot()

    #-----------------> Checking autoregressive order (p) of the Arima model with partial autocorrelation
    fig, axes = plt.subplots(1, 2, sharex=False)
    axes[0].plot(df3['Sales'].diff()); axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,5))
    plot_pacf(df3['Sales'].diff().dropna(), ax=axes[1])
    
    # #split 75%/25% training and test set
    # len_train = round(len(df3['Sales']) * 0.75) 
    # train = df3['Sales'][:len_train] 
    # test = df3['Sales'][len_train:]

    # #train model on train set
    # model = ARIMA(train, order=(1,1,1))
    # model_fit = model.fit()
    # print(model_fit.summary())

    # # Plot residual errors
    # residuals = pd.DataFrame(model_fit.resid)
    # fig, ax = plt.subplots(1,2)
    # residuals.plot(title="Residuals", ax=ax[0])
    # residuals.plot(kind='kde', title='Density', ax=ax[1])
    # plt.draw()

    # #forecast to compare with test set
    # fc_series =  model_fit.forecast(len(test), alpha=0.05)  # 95% conf

    # # Plot
    # plt.figure(figsize=(12,5), dpi=100)
    # plt.plot(train, label='training')
    # plt.plot(test, label='actual')
    # plt.plot(fc_series, label='forecast')
    # plt.title('Forecast vs Actuals')
    # plt.legend(loc='upper left', fontsize=8)
    # plt.show()
    
    #auto model, predict
    df_model = df3
    model_auto = pm.auto_arima(df_model['Sales'], start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=5, max_q=5, # maximum p and q
                        m=12,             # frequency = 1 year
                        d=None,           # let model determine 'd'
                        seasonal=True,    # Seasonality (SARIMA)
                        start_P=0, 
                        D=1, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=False, 
                        stepwise=True)

    print(model_auto.summary())
    model_auto.plot_diagnostics(figsize=(7,5))
    plt.show()

    # Forecast
    n_periods = 12
    fc, confint = model_auto.predict(n_periods=n_periods, return_conf_int=True)
    # make series for plotting purpose
    lower_series = pd.Series(confint[:, 0], index=fc.index)
    upper_series = pd.Series(confint[:, 1], index=fc.index)

    # Plot
    plt.plot(df_model['Sales'])
    plt.plot(fc, color='darkgreen')
    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)

    plt.title("Final Forecast +1 year")
    plt.show()

    # Actual vs Fitted
    # df3['predict'] = model_fit.predict()
    # df3.plot()

    # #original
    # fig, axes = plt.subplots(3, 2, sharex=False)
    # axes[0, 0].plot(df['Sales']); axes[0, 0].set_title('Original Series')
    # plot_acf(df['Sales'], ax=axes[0, 1])

    # # 1st Differencing
    # axes[1, 0].plot(df['Sales'].diff()); axes[1, 0].set_title('1st Order Differencing')
    # plot_acf(df['Sales'].diff().dropna(), ax=axes[1, 1])

    # # 2nd Differencing
    # axes[2, 0].plot(df['Sales'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    # plot_acf(df['Sales'].diff().diff().dropna(), ax=axes[2, 1])

    # plt.show()

    plot_others = False
    if plot_others:
        plt.figure(figsize=(20,5))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=75))
        df.groupby("Order Date YearMonth")["Sales"].sum().plot()
        plt.title("Global Superstore Monthly Sales")
        plt.legend(loc="best")
        plt.ylabel("Sales")
        plt.tight_layout()
        plt.show()

        g = sns.displot(df, x="Sales", kde=True, hue="Region")
        g.fig.subplots_adjust(top=.95)
        g.ax.set_title("Sales distribution")
        g.figure.autofmt_xdate()
        plt.show()

        sns.lineplot(x='Order Month', y='Sales', hue='Order Year', data=df)
        plt.show()

        df = df.loc[df['Category'] == "Technology"]

        df_orders = df.groupby(['Order ID', 'Region', 'Order Year', 'Order Month'], as_index=False)[['Sales']].sum()
        eda(df_orders, "Region", "Orders")

        for sc in df['Sub-Category'].unique():
            print(f"Preparing chart for product {sc}")
            df_product = df.loc[df['Sub-Category'] == sc]
            print(df_product.describe())
            q = df_product["Sales"].quantile(0.95)
            df_product = df_product.loc[df_product["Sales"] < q]
            print(f"Sub-Category {df_product.describe()}, rows: {len(df_product)}")
            eda(df_product, "Region", sc)
    
    plt.show()